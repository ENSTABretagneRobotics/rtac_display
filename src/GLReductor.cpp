#include <rtac_display/GLReductor.h>

namespace rtac { namespace display {

const std::string GLReductor::MainShader = std::string(R"(
#version 430 core

#define BLOCK_SIZE 256

layout(local_size_x = BLOCK_SIZE, local_size_y = 1) in;

layout(location = 0) uniform unsigned int N;

layout(std430, binding = 0) buffer inputBuffer
{
    Typename inputData[];
};
layout(std430, binding = 1) buffer outputBuffer
{
    Typename outputData[];
};

shared Typename s[BLOCK_SIZE];

Typename neutral(Typename initial);
Typename operator(Typename lhs, Typename rhs);

void main() 
{
    unsigned int idx      = gl_GlobalInvocationID.x;
    unsigned int gridSize = gl_NumWorkGroups.x*gl_WorkGroupSize.x;

    s[gl_LocalInvocationID.x] = neutral(inputData[idx]);
    while(idx < N) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x],
                                             inputData[idx]);
        idx += gridSize;
    }
    barrier();
    
    if(gl_LocalInvocationID.x < 128) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x],
                                             s[gl_LocalInvocationID.x + 128]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 64) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x], 
                                             s[gl_LocalInvocationID.x + 64]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 32) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x], 
                                             s[gl_LocalInvocationID.x + 32]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 16) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x], 
                                             s[gl_LocalInvocationID.x + 16]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 8) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x],
                                             s[gl_LocalInvocationID.x + 8]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 4) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x],
                                             s[gl_LocalInvocationID.x + 4]);
        barrier();
    }
    if(gl_LocalInvocationID.x < 2) {
        s[gl_LocalInvocationID.x] = operator(s[gl_LocalInvocationID.x],
                                             s[gl_LocalInvocationID.x + 2]);
        barrier();
    }

    if(gl_LocalInvocationID.x == 0) {
        outputData[gl_LocalInvocationID.x + gl_WorkGroupID.x] 
            = operator(s[gl_LocalInvocationID.x],
                       s[gl_LocalInvocationID.x + 1]);
    }
}

)");

const std::string GLReductor::SumOperatorShader = std::string(R"(
#version 430

Typename neutral(Typename initial)
{
    return initial*0;
}

Typename operator(Typename lhs, Typename rhs) {
    return lhs + rhs;
}
)");

const std::string GLReductor::SubOperatorShader = std::string(R"(
#version 430

Typename neutral(Typename initial)
{
    return initial*0;
}

Typename operator(Typename lhs, Typename rhs) {
    return lhs - rhs;
}
)");

const std::string GLReductor::MinOperatorShader = std::string(R"(
#version 430

Typename neutral(Typename initial)
{
    return initial;
}

Typename operator(Typename lhs, Typename rhs) {
    return min(lhs, rhs);
}
)");

const std::string GLReductor::MaxOperatorShader = std::string(R"(
#version 430

Typename neutral(Typename initial)
{
    return initial;
}
Typename operator(Typename lhs, Typename rhs) {
    return max(lhs, rhs);
}
)");


GLReductor::~GLReductor()
{
    for(auto program : programs_) {
        glDeleteProgram(program.second);
    }
}

GLuint GLReductor::add_program(const std::string& glslType, const std::string& op,
                               const std::string& operatorShader) const
{
    if(this->program(glslType, op)) {
        std::ostringstream oss;
        oss << "Program name '" << this->key(glslType, op) << "' already exists.";
        throw std::runtime_error(oss.str());
    }

    std::regex typenameRegex(Typename);

    // Instanciating shader template (much like c++ template instanciating.
    // Replacing Typename with glslType string).
    std::ostringstream mainShaderInstance, operatorShaderInstance;
    mainShaderInstance     << std::regex_replace(MainShader,     typenameRegex, glslType);
    operatorShaderInstance << std::regex_replace(operatorShader, typenameRegex, glslType);

    auto program = create_compute_program({operatorShaderInstance.str(),
                                           mainShaderInstance.str()});
    programs_[this->key(glslType, op)] = program;
    
    return program;
}

}; //namespace display
}; //namespace rtac
