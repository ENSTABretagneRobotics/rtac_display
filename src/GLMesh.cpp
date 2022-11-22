#include <rtac_display/GLMesh.h>

namespace rtac { namespace display {

const unsigned int GLMesh::GroupSize = 128;

const std::string GLMesh::expandVerticesShader = std::string(R"(
#version 430 core

layout(std430, binding=0) buffer inputData  { float inputAtt[]; };
layout(std430, binding=1) buffer inputFaces { uint  f[];   };

layout(std430, binding=2) buffer outputPoints  { float outputAtt[]; };

layout(location=0) uniform uint numFaces;
layout(location=1) uniform uint numComponents;

#define GROUP_SIZE 128

layout (local_size_x = GROUP_SIZE, local_size_y = 1) in;

void main()
{
    uint idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    for(; idx < numFaces; idx += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        uint i   = 3*idx;
        uint i0 = numComponents*f[i];
        uint i1 = numComponents*f[i + 1];
        uint i2 = numComponents*f[i + 2];

        uint iout = numComponents*i;
        for(int n = 0; n < numComponents; n++) {
            outputAtt[iout + n]                   = inputAtt[i0 + n];
            outputAtt[iout + n +   numComponents] = inputAtt[i1 + n];
            outputAtt[iout + n + 2*numComponents] = inputAtt[i2 + n];
        }
    }
}


)");

const std::string GLMesh::computeNormalsShader = std::string(R"(
#version 430 core

layout(std430, binding=0) buffer inputPoints  { float points[];  };
layout(std430, binding=1) buffer inputNormals { float normals[]; };

layout(location=0) uniform uint numPointSets;

#define GROUP_SIZE 128

layout (local_size_x = GROUP_SIZE, local_size_y = 1) in;

void main()
{
    uint idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    for(; idx < numPointSets; idx += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        uint i  = 9*idx;
        vec3 p0 = vec3(points[i]    , points[i + 1], points[i + 2]);
        vec3 p1 = vec3(points[i + 3], points[i + 4], points[i + 5]);
        vec3 p2 = vec3(points[i + 6], points[i + 7], points[i + 8]);

        vec3 n = normalize(cross(p1 - p0, p2 - p1));

        normals[i + 0] = n.x;
        normals[i + 1] = n.y;
        normals[i + 2] = n.z;
        normals[i + 3] = n.x;
        normals[i + 4] = n.y;
        normals[i + 5] = n.z;
        normals[i + 6] = n.x;
        normals[i + 7] = n.y;
        normals[i + 8] = n.z;
    }
}


)");

rtac::Bounds<float,3> GLMesh::bounding_box() const
{
    auto bbox = rtac::Bounds<float,3>::Zero();
    if(points_.size() == 0) {
        return bbox;
    }

    auto pointData = points_.map();
    bbox[0].init(pointData[0].x);
    bbox[1].init(pointData[0].y);
    bbox[2].init(pointData[0].z);
    for(unsigned int i = 1; i < points_.size(); i++) {
        bbox[0].update(pointData[i].x);
        bbox[1].update(pointData[i].y);
        bbox[2].update(pointData[i].z);
    }
    return bbox;
}

}; //namespace display
}; //namespace rtac
