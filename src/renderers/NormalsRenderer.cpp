#include <rtac_display/renderers/NormalsRenderer.h>

namespace rtac { namespace display {

const std::string NormalsRenderer::vertexShader = std::string(R"(
#version 430 core

in vec3 point;

uniform mat4 view;
uniform vec4 color;

out vec4 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = color;
}
)");

const std::string NormalsRenderer::fragmentShader = std::string(R"(
#version 430 core

in vec4 c;
out vec4 outColor;

void main()
{
    outColor = c;
}
)");

#define GROUP_SIZE 128
const std::string NormalsRenderer::generateLineDataShader = std::string(R"(
#version 430 core

// cannot use vec3 for alignment issues
//layout(std430, binding=0) buffer points  { vec3 p[]; };
//layout(std430, binding=1) buffer normals { vec3 n[]; };
//layout(std430, binding=2) buffer lines   { vec3 l[]; };
layout(std430, binding=0) buffer points  { float p[]; };
layout(std430, binding=1) buffer normals { float n[]; };
layout(std430, binding=2) buffer lines   { float l[]; };

layout(location=0) uniform uint numPoints;
layout(location=1) uniform bool do_normalize;

#define GROUP_SIZE 128

layout (local_size_x = GROUP_SIZE, local_size_y = 1) in;

void main()
{
    uint idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    for(; idx < numPoints; idx += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        uint i = 3*idx;
        vec3 p0 = vec3(p[i], p[i+1], p[i+2]);
        vec3 n0 = vec3(n[i], n[i+1], n[i+2]);
        if(do_normalize) {
            n0 = normalize(n0);
        }
        vec3 p1 = p0 + n0;

        l[2*i]     = p0.x;
        l[2*i + 1] = p0.y;
        l[2*i + 2] = p0.z;
        l[2*i + 3] = p1.x;
        l[2*i + 4] = p1.y;
        l[2*i + 5] = p1.z;
    }
}


)");

const std::string NormalsRenderer::generateLineDataShader2 = std::string(R"(
#version 430 core

// here point and normal data are interleaved in input buffer

// cannot use vec3 for alignment issues
//layout(std430, binding=0) buffer inputData  { vec3 input[]; };
//layout(std430, binding=2) buffer lines      { vec3 l[]; };
layout(std430, binding=0) buffer inputData  { float input[]; };
layout(std430, binding=2) buffer lines      { float l[]; };

layout(location=0) uniform uint numPoints;
layout(location=1) uniform bool do_normalize;

#define GROUP_SIZE 128

layout (local_size_x = GROUP_SIZE, local_size_y = 1) in;

void main()
{
    uint idx = gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x;
    for(; idx < numPoints; idx += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
        uint i = 6*idx;
        vec3 p0 = vec3(input[i],   input[i+1], input[i+2]);
        vec3 n0 = vec3(input[i+3], input[i+4], input[i+5]);
        if(do_normalize) {
            n0 = normalize(n0);
        }
        vec3 p1 = p0 + n0;

        l[i]     = p0.x;
        l[i + 1] = p0.y;
        l[i + 2] = p0.z;
        l[i + 3] = p1.x;
        l[i + 4] = p1.y;
        l[i + 5] = p1.z;
    }
}


)");

NormalsRenderer::Ptr NormalsRenderer::Create(const GLContext::Ptr& context,
                                             const View::Ptr& view,
                                             const Color::RGBAf& color)
{
    return Ptr(new NormalsRenderer(context, view, color));
}

NormalsRenderer::NormalsRenderer(const GLContext::Ptr& context,
                                 const View::Ptr& view,
                                 const Color::RGBAf& color) :
    Renderer(context, vertexShader, fragmentShader, view),
    numPoints_(0),
    displayData_(0),
    generateLineProgram_(create_compute_program(generateLineDataShader)),
    generateLineProgram2_(create_compute_program(generateLineDataShader2)),
    color_(color)
{}

NormalsRenderer::Ptr NormalsRenderer::New(const View::Ptr& view,
                                          const Color::RGBAf& color)
{
    return Ptr(new NormalsRenderer(view, color));
}

NormalsRenderer::NormalsRenderer(const View::Ptr& view,
                                 const Color::RGBAf& color) :
    Renderer(vertexShader, fragmentShader, view),
    numPoints_(0),
    displayData_(0),
    generateLineProgram_(create_compute_program(generateLineDataShader)),
    generateLineProgram2_(create_compute_program(generateLineDataShader2)),
    color_(color)
{}

NormalsRenderer::~NormalsRenderer()
{
    this->delete_data();
}

void NormalsRenderer::allocate_data(size_t numPoints)
{
    if(!displayData_) {
        glGenBuffers(1, &displayData_);
    }
    if(numPoints > numPoints_) {
        glBindBuffer(GL_ARRAY_BUFFER, displayData_);
        // each point of the model needs 2 points to draw the normal
        glBufferData(GL_ARRAY_BUFFER, 6*sizeof(float)*numPoints, NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void NormalsRenderer::delete_data()
{
    if(displayData_ > 0) {
        glDeleteBuffers(1, &displayData_);
    }
    displayData_ = 0;
    numPoints_   = 0;
}

void NormalsRenderer::set_normals(size_t numPoints, GLuint points, GLuint normals,
                                  bool normalizeNormals)
{
    this->allocate_data(numPoints);

    glUseProgram(generateLineProgram_);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, points);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, normals);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, displayData_);

    glUniform1ui(0, numPoints);
    if(normalizeNormals)
        glUniform1ui(1, 1);
    else
        glUniform1ui(1, 0);

    glDispatchCompute((numPoints / GROUP_SIZE) + 1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    
    numPoints_ = numPoints;
}

void NormalsRenderer::set_normals(size_t numPoints, GLuint input, bool normalizeNormals)
{
    this->allocate_data(numPoints);

    glUseProgram(generateLineProgram2_);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, displayData_);

    glUniform1ui(0, numPoints);
    if(normalizeNormals)
        glUniform1ui(1, 1);
    else
        glUniform1ui(1, 0);

    glDispatchCompute((numPoints / GROUP_SIZE) + 1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(0);
    
    numPoints_ = numPoints;
}
void NormalsRenderer::set_pose(const Pose& pose)
{
    pose_ = pose;
}

void NormalsRenderer::set_color(const Color::RGBAf& color)
{
    color_.r = std::max(0.0f, std::min(1.0f, color.r));
    color_.g = std::max(0.0f, std::min(1.0f, color.g));
    color_.b = std::max(0.0f, std::min(1.0f, color.b));
    color_.a = std::max(0.0f, std::min(1.0f, color.a));
}

void NormalsRenderer::draw() const
{
    this->draw(this->view());
}

void NormalsRenderer::draw(const View::ConstPtr& view) const
{
    if(displayData_ == 0 || numPoints_ == 0)
        return;
    
    Mat4 fullView = view->view_matrix() * pose_.homogeneous_matrix();

    glUseProgram(renderProgram_);
    
    glBindBuffer(GL_ARRAY_BUFFER, displayData_);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, fullView.data());
    glUniform4fv(glGetUniformLocation(renderProgram_, "color"),
        1, reinterpret_cast<const float*>(&color_));

    glDrawArrays(GL_LINES, 0, 2*numPoints_);
    
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);
}

}; //namespace display
}; //namespace rtac

