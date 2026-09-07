#pragma once
// Terminal-only RGB readback sink. No decoded/motion/depth/SR intermediate
// enters this pipe; at most one packed frame + a 256 KiB kernel pipe is held.
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

class NativeSoftwareEncoder {
    HANDLE pipe_ = INVALID_HANDLE_VALUE, process_ = nullptr, event_ = nullptr;
    HANDLE child_job_ = nullptr;
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
    std::vector<UCHAR> hash_object_;
    bool finished_ = false;
    std::wstring cancel_;
    static std::wstring env(const wchar_t* name) {
        DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
        if (!size) return {};
        std::wstring value(size, L'\0');
        GetEnvironmentVariableW(name, value.data(), size);
        value.resize(size-1); return value;
    }
    // CommandLineToArgvW quoting, not a shell. Paths may contain spaces/Unicode.
    static std::wstring quote(const std::wstring& arg) {
        std::wstring out=L"\""; size_t slash=0;
        for(wchar_t ch:arg) {
            if(ch==L'\\') {++slash; continue;}
            out.append(slash*(ch==L'"'?2:1),L'\\'); slash=0;
            if(ch==L'"') out+=L'\\'; out+=ch;
        }
        out.append(slash*2,L'\\'); return out+L'"';
    }
    bool cancelled() const {
        return !cancel_.empty() && GetFileAttributesW(cancel_.c_str())!=INVALID_FILE_ATTRIBUTES;
    }
    bool complete(OVERLAPPED& ov, DWORD& transferred, DWORD seconds) {
        const ULONGLONG until=GetTickCount64()+seconds*1000ull;
        while(GetTickCount64()<until && !cancelled()) {
            if(WaitForSingleObject(event_,100)==WAIT_OBJECT_0)
                return GetOverlappedResult(pipe_,&ov,&transferred,FALSE)!=FALSE;
            if(process_ && WaitForSingleObject(process_,0)==WAIT_OBJECT_0) break;
        }
        CancelIoEx(pipe_,&ov);
        // The OVERLAPPED storage must remain alive until cancellation completes.
        GetOverlappedResult(pipe_,&ov,&transferred,TRUE);
        return false;
    }
public:
    std::string error, rgb_sha256;
    UINT64 rgb_bytes=0;
    ~NativeSoftwareEncoder() {
        if(pipe_!=INVALID_HANDLE_VALUE) CloseHandle(pipe_);
        if(process_) {
            if(WaitForSingleObject(process_,0)!=WAIT_OBJECT_0) {
                TerminateProcess(process_,130); WaitForSingleObject(process_,5000);
            }
            CloseHandle(process_);
        }
        if(child_job_) CloseHandle(child_job_);
        if(event_) CloseHandle(event_);
        if(hash_) BCryptDestroyHash(hash_);
        if(alg_) BCryptCloseAlgorithmProvider(alg_,0);
    }
    bool init(const std::string& encoder, UINT width, UINT height,
              UINT fps_n, UINT fps_d, bool bt709) {
        if(encoder!="libx264" && encoder!="libx265" && encoder!="ffv1") {
            error="unsupported_terminal_software_encoder"; return false;
        }
        const auto exe=env(L"XESS_TERMINAL_FFMPEG"), output=env(L"XESS_TERMINAL_OUTPUT"),
                   log_path=env(L"XESS_TERMINAL_LOG");
        cancel_=env(L"XESS_TERMINAL_CANCEL");
        if(exe.empty()||output.empty()||log_path.empty()) {error="missing_terminal_paths";return false;}
        const auto pipe_name=L"\\\\.\\pipe\\xess-terminal-"+std::to_wstring(GetCurrentProcessId())+L"-"+std::to_wstring(GetTickCount64());
        pipe_=CreateNamedPipeW(pipe_name.c_str(),PIPE_ACCESS_OUTBOUND|FILE_FLAG_OVERLAPPED|FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE|PIPE_WAIT|PIPE_REJECT_REMOTE_CLIENTS,1,256*1024,0,0,nullptr);
        event_=CreateEventW(nullptr,TRUE,FALSE,nullptr);
        if(pipe_==INVALID_HANDLE_VALUE||!event_) {error="terminal_pipe_create";return false;}
        SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
        HANDLE log=CreateFileW(log_path.c_str(),GENERIC_WRITE,FILE_SHARE_READ,&sa,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        HANDLE nul=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_EXISTING,0,nullptr);
        if(log==INVALID_HANDLE_VALUE||nul==INVALID_HANDLE_VALUE) {
            if(log!=INVALID_HANDLE_VALUE)CloseHandle(log);if(nul!=INVALID_HANDLE_VALUE)CloseHandle(nul);
            error="terminal_log_open"; return false;
        }
        std::vector<std::wstring> args={exe,L"-hide_banner",L"-v",L"error",L"-nostdin",L"-n",L"-f",L"rawvideo",
            L"-pixel_format",L"rgb24",L"-video_size",std::to_wstring(width)+L"x"+std::to_wstring(height),
            L"-framerate",std::to_wstring(fps_n)+L"/"+std::to_wstring(fps_d),L"-i",pipe_name,
            L"-an",L"-fps_mode",L"passthrough",L"-c:v",std::wstring(encoder.begin(),encoder.end())};
        if(encoder=="ffv1") {
            args.insert(args.end(),{L"-level",L"3",L"-pix_fmt",L"bgr0",L"-color_range",L"pc"});
        } else {
            const std::wstring matrix=bt709?L"bt709":L"bt601";
            args.insert(args.end(),{L"-vf",L"scale=in_range=full:out_range=limited:out_color_matrix="+matrix+L",format=yuv420p",
                L"-preset",L"medium",L"-crf",L"18",L"-bf",L"0",L"-color_range",L"tv",L"-colorspace",bt709?L"bt709":L"smpte170m"});
        }
        args.insert(args.end(),{L"-f",L"matroska",output});
        std::wstring command;for(const auto& arg:args) {if(!command.empty())command+=L' ';command+=quote(arg);}
        STARTUPINFOW si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;
        si.hStdOutput=si.hStdError=log;si.hStdInput=nul;PROCESS_INFORMATION pi{};
        // A native crash must not orphan the FFmpeg child. The handle is not
        // inherited, so Windows also terminates it if this process is killed.
        child_job_=CreateJobObjectW(nullptr,nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
        limit.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if(!child_job_ || !SetInformationJobObject(child_job_,JobObjectExtendedLimitInformation,&limit,sizeof(limit))) {
            CloseHandle(log);CloseHandle(nul);error="terminal_child_job_create";return false;
        }
        const BOOL started=CreateProcessW(exe.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,nullptr,&si,&pi);
        CloseHandle(log);CloseHandle(nul);
        if(!started) {error="terminal_ffmpeg_start="+std::to_string(GetLastError());return false;}
        process_=pi.hProcess;
        if(!AssignProcessToJobObject(child_job_,process_) || ResumeThread(pi.hThread)==static_cast<DWORD>(-1)) {
            TerminateProcess(process_,130);CloseHandle(pi.hThread);error="terminal_child_job_assign";return false;
        }
        CloseHandle(pi.hThread);
        OVERLAPPED ov{};ov.hEvent=event_;DWORD transferred=0;
        const BOOL connected=ConnectNamedPipe(pipe_,&ov);
        const DWORD status=connected?ERROR_SUCCESS:GetLastError();
        if(status!=ERROR_SUCCESS && status!=ERROR_PIPE_CONNECTED &&
           !(status==ERROR_IO_PENDING && complete(ov,transferred,30))) {
            error="terminal_ffmpeg_connect_failed; see terminal.log";return false;
        }
        if(encoder=="ffv1") {
            DWORD size=0,read=0;
            if(BCryptOpenAlgorithmProvider(&alg_,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0 ||
               BCryptGetProperty(alg_,BCRYPT_OBJECT_LENGTH,(PUCHAR)&size,sizeof(size),&read,0)<0) {
                error="terminal_hash_init";return false;
            }
            hash_object_.resize(size);
            if(BCryptCreateHash(alg_,&hash_,hash_object_.data(),size,nullptr,0,0)<0) {error="terminal_hash_create";return false;}
        }
        return true;
    }
    bool write(const UCHAR* data, size_t bytes) {
        if(finished_||cancelled()) {error="terminal_cancelled_or_closed";return false;}
        size_t offset=0;
        while(offset<bytes) {
            OVERLAPPED ov{};ov.hEvent=event_;ResetEvent(event_);DWORD count=0;
            const DWORD chunk=static_cast<DWORD>(std::min<size_t>(bytes-offset,256*1024));
            BOOL ok=WriteFile(pipe_,data+offset,chunk,&count,&ov);
            if(!ok && GetLastError()==ERROR_IO_PENDING) ok=complete(ov,count,60);
            if(!ok||!count) {error="terminal_encoder_pipe_closed_or_timeout; see terminal.log";return false;}
            offset+=count;
        }
        if(hash_ && BCryptHashData(hash_,const_cast<PUCHAR>(data),static_cast<ULONG>(bytes),0)<0) {
            error="terminal_hash_update";return false;
        }
        rgb_bytes+=bytes;return true;
    }
    bool finish() {
        if(finished_) return error.empty();
        finished_=true;
        // Writes completed before closing; do not DisconnectNamedPipe (that
        // discards unread bytes) or FlushFileBuffers (unbounded consumer wait).
        if(pipe_!=INVALID_HANDLE_VALUE) {CloseHandle(pipe_);pipe_=INVALID_HANDLE_VALUE;}
        const ULONGLONG until=GetTickCount64()+120000;
        while(GetTickCount64()<until && !cancelled()) {
            if(WaitForSingleObject(process_,100)==WAIT_OBJECT_0) {
                DWORD rc=1;GetExitCodeProcess(process_,&rc);
                if(rc) {error="terminal_ffmpeg_exit="+std::to_string(rc)+"; see terminal.log";return false;}
                if(hash_) {
                    UCHAR digest[32];if(BCryptFinishHash(hash_,digest,32,0)<0) {error="terminal_hash_finish";return false;}
                    const char* hex="0123456789abcdef";for(UCHAR byte:digest) {rgb_sha256+=hex[byte>>4];rgb_sha256+=hex[byte&15];}
                }
                return true;
            }
        }
        error="terminal_ffmpeg_finish_timeout_or_cancelled";return false;
    }
};
