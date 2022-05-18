#include <rtac_display/GLMesh.h>

namespace rtac { namespace display {

const unsigned int GLMesh::GroupSize = 128;

const std::string GLMesh::computeNormalsShader = std::string(R"(
#version 430 core

layout(std430, binding=0) buffer inputPoints { float pin[]; };
layout(std430, binding=1) buffer inputFaces  { uint  f[]; };

layout(std430, binding=2) buffer outputPoints  { float pout[]; };
layout(std430, binding=3) buffer outputNormals { float nout[]; };

layout(location=0) uniform uint numFaces;

#define GROUP_SIZE 128

layout (local_size_x = GROUP_SIZE, local_size_y = 1) in;

void main()
{
    uint idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    for(; idx < numFaces; idx += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        uint i   = 3*idx;
        uint ip0 = 3*f[i];
        uint ip1 = 3*f[i + 1];
        uint ip2 = 3*f[i + 2];

        vec3 p0 = vec3(pin[ip0], pin[ip0 + 1], pin[ip0 + 2]);
        vec3 p1 = vec3(pin[ip1], pin[ip1 + 1], pin[ip1 + 2]);
        vec3 p2 = vec3(pin[ip2], pin[ip2 + 1], pin[ip2 + 2]);

        vec3 n = normalize(cross(p1 - p0, p2 - p1));
        
        uint iout = 3*i;
        pout[iout]     = p0.x;
        pout[iout + 1] = p0.y;
        pout[iout + 2] = p0.z;
        pout[iout + 3] = p1.x;
        pout[iout + 4] = p1.y;
        pout[iout + 5] = p1.z;
        pout[iout + 6] = p2.x;
        pout[iout + 7] = p2.y;
        pout[iout + 8] = p2.z;
        nout[iout]     = n.x;
        nout[iout + 1] = n.y;
        nout[iout + 2] = n.z;
        nout[iout + 3] = n.x;
        nout[iout + 4] = n.y;
        nout[iout + 5] = n.z;
        nout[iout + 6] = n.x;
        nout[iout + 7] = n.y;
        nout[iout + 8] = n.z;
    }
}


)");

}; //namespace display
}; //namespace rtac
