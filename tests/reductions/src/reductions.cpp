#include "reductions.h"

namespace rtac { namespace display {

const std::string elementfShader = std::string( R"(
#version 430 core

layout(local_size_x = 1, local_size_y = 1) in;
//layout(std140, binding = 0) buffer input
layout(std430, binding = 0) buffer input
{
    float data[];
};

void main() {
    data[gl_GlobalInvocationID.x] = 1;
}

)");

const std::string element3fShader = std::string( R"(
#version 430 core

layout(local_size_x = 1, local_size_y = 1) in;
//layout(std140, binding = 0) buffer input
layout(std430, binding = 0) buffer input
{
    vec3 data[];
};

void main() {
    data[gl_GlobalInvocationID.x].x = 1;
    data[gl_GlobalInvocationID.x].y = 0;
    data[gl_GlobalInvocationID.x].z = 0;
}
)");

const std::string element4fShader = std::string( R"(
#version 430 core

layout(local_size_x = 1, local_size_y = 1) in;
//layout(std140, binding = 0) buffer input
layout(std430, binding = 0) buffer input
{
    vec4 data[];
};

void main() {
    data[gl_GlobalInvocationID.x].x = 1;
    data[gl_GlobalInvocationID.x].y = 0;
    data[gl_GlobalInvocationID.x].z = 0;
    data[gl_GlobalInvocationID.x].w = 1;
}
)");

void element(const GLVector<float>& data)
{
    static const GLuint program = create_compute_program(elementfShader);
    //static const GLuint program = create_compute_program(element4fShader);
    
    // fail as it should (alignement issues)
    // static const GLuint program = create_compute_program(element3fShader); 
    
    glUseProgram(program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, data.gl_id());
    
    glDispatchCompute(data.size(), 1, 1);
    //glDispatchCompute(data.size() / 3, 1, 1);
    //glDispatchCompute(data.size() / 4, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    data.unbind(GL_SHADER_STORAGE_BUFFER);

    glUseProgram(0);

    GL_CHECK_LAST();
}



const std::string sumShaderf = std::string( R"(
#version 430 core

layout(local_size_x = 256, local_size_y = 1) in;
layout(location = 0) uniform unsigned int N;
layout(std430, binding = 0) buffer inputBuffer
{
    float inputData[];
};
layout(std430, binding = 1) buffer outputBuffer
{
    float outputData[];
};

shared float s[256];

void main() 
{
    unsigned int idx      = gl_GlobalInvocationID.x;
    unsigned int gridSize = gl_NumWorkGroups.x*gl_WorkGroupSize.x;

    s[gl_LocalInvocationID.x] = 0;
    while(idx < N) {
        s[gl_LocalInvocationID.x] += inputData[idx];
        idx += gridSize;
    }
    barrier();

    if(gl_LocalInvocationID.x < 128) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 128];
        barrier();
    }
    if(gl_LocalInvocationID.x < 64) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 64];
        barrier();
    }
    if(gl_LocalInvocationID.x < 32) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 32];
        barrier();
    }
    if(gl_LocalInvocationID.x < 16) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 16];
        barrier();
    }
    if(gl_LocalInvocationID.x < 8) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 8];
        barrier();
    }
    if(gl_LocalInvocationID.x < 4) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 4];
        barrier();
    }
    if(gl_LocalInvocationID.x < 2) {
        s[gl_LocalInvocationID.x] += s[gl_LocalInvocationID.x + 2];
        barrier();
    }

    if(gl_LocalInvocationID.x == 0) {
        outputData[gl_LocalInvocationID.x + gl_WorkGroupID.x] 
            = s[gl_LocalInvocationID.x] + s[gl_LocalInvocationID.x + 1];
    }
}

)");

const std::string reductionShaderf = std::string( R"(
#version 430 core

layout(local_size_x = 256, local_size_y = 1) in;
layout(location = 0) uniform unsigned int N;
layout(std430, binding = 0) buffer inputBuffer
{
    float inputData[];
};
layout(std430, binding = 1) buffer outputBuffer
{
    float outputData[];
};

shared float s[256];

float operator(float lhs, float rhs);
//{
//    return lhs + rhs;
//}

void main() 
{
    unsigned int idx      = gl_GlobalInvocationID.x;
    unsigned int gridSize = gl_NumWorkGroups.x*gl_WorkGroupSize.x;

    s[gl_LocalInvocationID.x] = 0;
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

const std::string reductionShader4f = std::string( R"(
#version 430 core

layout(local_size_x = 256, local_size_y = 1) in;
layout(location = 0) uniform unsigned int N;
layout(std430, binding = 0) buffer inputBuffer
{
    vec4 inputData[];
};
layout(std430, binding = 1) buffer outputBuffer
{
    vec4 outputData[];
};

shared vec4 s[256];

vec4 operator(vec4 lhs, vec4 rhs);

void main() 
{
    unsigned int idx      = gl_GlobalInvocationID.x;
    unsigned int gridSize = gl_NumWorkGroups.x*gl_WorkGroupSize.x;

    s[gl_LocalInvocationID.x] = vec4(0,0,0,0);
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

const std::string plusOperatorShader = std::string(R"(
#version 430 core

float operator(float lhs, float rhs)
{
    return lhs + rhs;
}

vec4 operator(vec4 lhs, vec4 rhs)
{
    return lhs + rhs;
}

)");

const std::string maxOperatorShader = std::string(R"(
#version 430 core

float operator(float lhs, float rhs)
{
    return max(lhs, rhs);
}

vec4 operator(vec4 lhs, vec4 rhs)
{
    return max(lhs, rhs);
}

)");


float sum(const GLVector<float>& data)
{
    //static const GLuint program = create_compute_program(sumShaderf);
    //static const GLuint program = create_compute_program(reductionShaderf);
    // static const GLuint program = create_compute_program({
    //     //plusOperatorShader,
    //     maxOperatorShader,
    //     reductionShaderf});
    // unsigned int N = data.size();

    static const GLuint program = create_compute_program({
        //plusOperatorShader,
        maxOperatorShader,
        reductionShader4f});
    unsigned int N = data.size() / 4;

    glUseProgram(program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, data.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, data.gl_id());
    
    while(N > 0) {
        unsigned int blockCount = N / (2*256);

        glUniform1ui(0, N);
        if(blockCount == 0)
            glDispatchCompute(1,1,1);
        else
            glDispatchCompute(blockCount,1,1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        N = blockCount;
    }

    data.unbind(GL_SHADER_STORAGE_BUFFER);

    glUseProgram(0);

    GL_CHECK_LAST();

    return 0.0;
}

}; //namespace display
}; //namespace rtac




