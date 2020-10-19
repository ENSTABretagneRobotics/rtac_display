#include <optix_helpers/display/MeshRenderer.h>

namespace optix_helpers { namespace display {

const std::string MeshRendererObj::vertexShader = std::string( R"(
#version 430 core

in vec3 point;
in vec3 normal;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 color;

out vec3 c;

void main()
{
    gl_Position = view*vec4(point, 1.0f);
    c = vec3(1.0,1.0,1.0) * abs(dot(mat3(view)*normal, normalize(vec3(gl_Position))));
    gl_Position = projection * gl_Position;
    //c = vec3(1.0,1.0,1.0);
}
)");

const std::string MeshRendererObj::fragmentShader = std::string(R"(
#version 430 core

in vec3 c;
out vec4 outColor;

void main()
{
    outColor = vec4(c, 1.0f);
}
)");

MeshRendererObj::MeshRendererObj(const View3D& view, const Color& color) :
    RendererObj(vertexShader, fragmentShader, view),
    numPoints_(0),
    points_(0),
    normals_(0),
    color_(color)
{
}

void MeshRendererObj::allocate_points(size_t numPoints)
{
    if(!points_) {
        glGenBuffers(1, &points_);
    }
    if(!normals_) {
        glGenBuffers(1, &normals_);
    }
    if(numPoints_ < numPoints) {
        glBindBuffer(GL_ARRAY_BUFFER, points_);
        glBufferData(GL_ARRAY_BUFFER, 3*sizeof(float)*numPoints, NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, normals_);
        glBufferData(GL_ARRAY_BUFFER, 3*sizeof(float)*numPoints, NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void MeshRendererObj::delete_points()
{
    if(points_ > 0) {
        glDeleteBuffers(1, &points_);
    }
    points_ = 0;
    if(normals_ > 0) {
        glDeleteBuffers(1, &normals_);
    }
    normals_ = 0;
    numPoints_ = 0;
}

void MeshRendererObj::set_mesh(const Mesh& mesh)
{
    using namespace rtac::types;
    using namespace rtac::types::indexing;

    this->allocate_points(3*mesh.num_faces());
    
    Map<const Matrix<float>>    mPoints = mesh.points();
    Map<const Matrix<uint32_t>> mFaces  = mesh.faces();

    glBindBuffer(GL_ARRAY_BUFFER, points_);
    auto points  = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    glBindBuffer(GL_ARRAY_BUFFER, normals_);
    auto normals = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));

    for(int nf = 0; nf < mesh.num_faces(); nf++) {
        Vector3<uint32_t> f = mFaces(nf, all);
        Vector3<float> p0 = mPoints(f(0), all);
        Vector3<float> p1 = mPoints(f(1), all);
        Vector3<float> p2 = mPoints(f(2), all);
        Vector3<float> n = ((p1 - p0).cross(p2 - p0)).normalized();
        
        int i = 9*nf;
        points[i]     = p0(0); points[i + 1] = p0(1); points[i + 2] = p0(2);
        points[i + 3] = p1(0); points[i + 4] = p1(1); points[i + 5] = p1(2);
        points[i + 6] = p2(0); points[i + 7] = p2(1); points[i + 8] = p2(2);
        normals[i]     = n(0); normals[i + 1] = n(1); normals[i + 2] = n(2);
        normals[i + 3] = n(0); normals[i + 4] = n(1); normals[i + 5] = n(2);
        normals[i + 6] = n(0); normals[i + 7] = n(1); normals[i + 8] = n(2);
    }
    
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    numPoints_ = 9*mesh.num_faces();
}

void MeshRendererObj::set_pose(const Pose& pose)
{
    pose_ = pose;
}

void MeshRendererObj::set_color(const Color& color)
{
    color_[0] = std::max(0.0f, std::min(1.0f, color[0]));
    color_[1] = std::max(0.0f, std::min(1.0f, color[1]));
    color_[2] = std::max(0.0f, std::min(1.0f, color[2]));
}

void MeshRendererObj::draw()
{
    if(points_ == 0 || normals_ == 0|| numPoints_ == 0)
        return;

    GLfloat pointSize;
    glGetFloatv(GL_POINT_SIZE, &pointSize);
    glPointSize(3);

    glEnable(GL_DEPTH_TEST);

    glUseProgram(renderProgram_);
    
    glBindBuffer(GL_ARRAY_BUFFER, points_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, normals_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    
    auto view = view_.downcast<View3DObj>();
    Mat4 viewMatrix = view->raw_view_matrix() * pose_.homogeneous_matrix();
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, viewMatrix.data());
    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "projection"),
        1, GL_FALSE, view->projection_matrix().data());
    glUniform3fv(glGetUniformLocation(renderProgram_, "color"),
        1, color_.data());

    glDrawArrays(GL_TRIANGLES, 0, numPoints_);

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(0);

    glPointSize(pointSize);
}

}; //namespace display
}; //namespace optix_helpers


