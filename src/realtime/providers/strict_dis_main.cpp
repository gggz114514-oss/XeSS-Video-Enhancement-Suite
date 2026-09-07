// Compile-time adapter only: A owns decode/depth/Mask/SR/FG/post/encode.
// The build fixes NATIVE_GPU_CORE_ROOT to an audited source snapshot.
#define XESS_NATIVE_GPU_CORE_LIBRARY
#include <vpl_gpu_full_fg.cpp>
#include "../native_strict_dis_provider.h"

namespace {
std::string dis_shader_directory;
uint64_t dis_record_count=0,dis_shader_count=0;
class TrackedStrictDis final : public xess_gpu::MotionProvider {
    std::unique_ptr<xess_gpu::MotionProvider> inner;
public:
    TrackedStrictDis():inner(xess_gpu::make_strict_dis_provider(dis_shader_directory,
        xess_gpu::StrictDisPreset::Medium,xess_gpu::StrictDisInput::Ffmpeg71RgbGrayLimited)) {}
    const char* backend_name() const noexcept override { return inner->backend_name(); }
    bool initialize(ID3D12Device* d,const xess_gpu::ProviderConfig& c,std::string& e) override { return inner->initialize(d,c,e); }
    bool record(const xess_gpu::FrameLease* p,const xess_gpu::FrameLease& c,uint32_t s,
                xess_gpu::RecordContext& r,xess_gpu::MotionPacket& m,xess_gpu::Counters& n,std::string& e) override {
        const bool ok=inner->record(p,c,s,r,m,n,e);
        if(ok) { ++dis_record_count; dis_shader_count=xess_gpu::strict_dis_shader_dispatches(*inner); }
        return ok;
    }
};
}
int main(int argc,char** argv) {
    std::vector<char*> args; args.push_back(argv[0]); std::string provider_report;
    char module[MAX_PATH]{}; GetModuleFileNameA(nullptr,module,MAX_PATH);
    dis_shader_directory=module;
    dis_shader_directory=dis_shader_directory.substr(0,dis_shader_directory.find_last_of("\\/"));
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"--dis-shader-dir") && i+1<argc) dis_shader_directory=argv[++i];
        else if(!std::strcmp(argv[i],"--dis-provider-report") && i+1<argc) provider_report=argv[++i];
        else args.push_back(argv[i]);
    }
    xess_gpu::native_motion_factory=[](const char* backend)->xess_gpu::MotionProvider* {
        return std::strcmp(backend,"gpu-dis")==0?new TrackedStrictDis():nullptr;
    };
    const int result=xess_native_gpu_main(static_cast<int>(args.size()),args.data());
    if(!provider_report.empty()) {
        FILE* out=std::fopen(provider_report.c_str(),"wb");
        if(!out) return result?result:3;
        std::fprintf(out,"{\"preset\":\"MEDIUM\",\"input_profile\":\"FFmpeg7.1_RGB24_cv2GRAY_limited\",\"records\":%llu,\"internal_shader_dispatches\":%llu,\"native_exit_code\":%d}\n",
                     dis_record_count,dis_shader_count,result);
        std::fclose(out);
    }
    return result;
}
