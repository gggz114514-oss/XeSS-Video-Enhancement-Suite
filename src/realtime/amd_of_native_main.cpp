#if !__has_include("vpl_gpu_full_fg.cpp")
#error AMD worker entry requires the private native host; use the published R4 binary.
#else
#define XESS_NATIVE_GPU_CORE_LIBRARY
#include "vpl_gpu_full_fg.cpp"
#include "amd_of_provider.h"

namespace {
std::string amd_shader_directory;
AmdProviderStatistics amd_statistics;
xess_gpu::MotionProvider* create_amd_provider(const char* backend){
    if(!backend||std::strcmp(backend,"amd-of"))return nullptr;
    return new AmdMotionProvider(amd_shader_directory,amd_statistics);
}
}

int main(int argc,char** argv){
    std::string report;bool backend_set=false;
    std::vector<std::string> arguments;
    for(int i=0;i<argc;++i){
        arguments.emplace_back(argv[i]);
        if(!std::strcmp(argv[i],"--shader-dir")&&i+1<argc)amd_shader_directory=argv[i+1];
        if(!std::strcmp(argv[i],"--report")&&i+1<argc)report=argv[i+1];
        if(!std::strcmp(argv[i],"--motion-backend")){
            backend_set=true;
            if(i+1>=argc||std::strcmp(argv[i+1],"amd-of")){
                std::fprintf(stderr,"[amd-of] this entry only accepts amd-of; no fallback backend\n");return 2;
            }
        }
    }
    if(!backend_set){arguments.emplace_back("--motion-backend");arguments.emplace_back("amd-of");}
    std::vector<char*> pointers;for(auto& argument:arguments)pointers.push_back(argument.data());
    xess_gpu::native_motion_factory=create_amd_provider;
    const int result=xess_native_gpu_main(static_cast<int>(pointers.size()),pointers.data());
    if(!report.empty()){
        std::ofstream output(report+".amd.json",std::ios::binary);
        const auto& s=amd_statistics;
        output<<"{\n  \"schema\": \"amd-native-provider-v1\", \"backend\": \"amd-of\", \"returncode\": "<<result
            <<",\n  \"frames\": "<<s.frames<<", \"source_pair_analysis_count\": "<<s.pairs
            <<", \"sdk_directional_call_count\": "<<s.sdk_calls<<", \"forward_sdk_calls\": "<<s.forward_calls
            <<", \"reverse_sdk_calls\": "<<s.reverse_calls<<", \"first_frame_self_analysis_count\": "<<s.first_self_calls
            <<", \"dense_adapter_dispatch_count\": "<<s.dense_dispatches<<", \"history_reset_count\": "<<s.history_resets
            <<", \"official_warmup_pairs\": "<<s.warmup_pairs<<", \"retired_slot_reuses\": "<<s.reused_slots
            <<", \"provider_full_frame_cpu_bytes\": 0, \"duplicate_rgba_conversion_count\": 0, \"provider_submit_count\": 0, \"provider_cpu_wait_count\": 0,"
            <<"\n  \"sdk_commit\": \"60f4ea81909200d8542eca14dccb2628b763a9a3\", \"shader_configuration\": \"official_FP32_cs6.2\","
            <<"\n  \"flow_contract\": \"R16G16_SINT ceil8 grid, current-to-previous source pixels; bilinear source R32G32_FLOAT\","
            <<"\n  \"confidence\": \"unavailable from AMD; common core cycle/photometric\", \"mask_tagged\": false,"
            <<"\n  \"warmup_policy\": \"official zero flow for context FrameIndex<=5; not reset each reverse pair\","
            <<"\n  \"scene_reset_policy\": \"explicit input reset immediately; core-detected scene cut conveyed by previous metadata at next source\","
            <<"\n  \"error\": \""<<json_escape(s.error)<<"\"\n}\n";
    }
    return result;
}
#endif
