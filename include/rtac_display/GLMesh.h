#ifndef _DEF_RTAC_DISPLAY_GL_MESH_H_
#define _DEF_RTAC_DISPLAY_GL_MESH_H_

#include <rtac_base/types/Handle.h>
#include <rtac_base/types/Point.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_display/GLFormat.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

class GLMesh
{
    public:

    using Ptr      = rtac::types::Handle<GLMesh>;
    using ConstPtr = rtac::types::Handle<const GLMesh>;

    using Point    = types::Point3<float>;
    using Face     = types::Point3<uint32_t>;
    using UV       = types::Point2<float>;
    using Normal   = types::Point3<float>;

    using BaseMesh = types::Mesh<Point,Face>;

    static const unsigned int GroupSize;
    static const std::string computeNormalsShader;

    protected:
    
    GLVector<Point>  points_;
    GLVector<Face>   faces_;
    GLVector<UV>     uvs_;
    GLVector<Normal> normals_;

    public:

    static Ptr Create() { return Ptr(new GLMesh()); }

    GLMesh() {}
    GLMesh(GLMesh&& other);
    GLMesh& operator=(GLMesh&& other);

    template <template <typename> class VectorT>
    GLMesh(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other);
    template <template <typename> class VectorT>
    GLMesh& operator=(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other);

    GLVector<Point>&  points()  { return points_; }
    GLVector<Face>&   faces()   { return faces_; }
    GLVector<UV>&     uvs()     { return uvs_; }
    GLVector<Normal>& normals() { return normals_; }

    const GLVector<Point>&  points()   const { return points_; }
    const GLVector<Face>&   faces()    const { return faces_; }
    const GLVector<UV>&     uvs()      const { return uvs_; }
    const GLVector<Normal>& normals()  const { return normals_; }

    static GLMesh::Ptr cube(float scale = 1.0f) {
        return Ptr(new GLMesh(BaseMesh::cube(scale)));
    }

    void compute_normals();
};

inline GLMesh::GLMesh(GLMesh&& other) :
    GLMesh()
{
    *this = std::move(other);
}

inline GLMesh& GLMesh::operator=(GLMesh&& other)
{
    points_   = std::move(other.points_);
    faces_    = std::move(other.faces_);
    uvs_      = std::move(other.uvs_);
    normals_  = std::move(other.normals_);

    return *this;
}

template <template <typename> class VectorT>
inline GLMesh::GLMesh(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other) :
    GLMesh()
{
    *this = other;
}

template <template <typename> class VectorT>
inline GLMesh& GLMesh::operator=(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other)
{
    points_ = other.points();
    faces_  = other.faces();

    uvs_.resize(0);
    normals_.resize(0);

    return *this;
}

inline void GLMesh::compute_normals()
{
    if(faces_.size() == 0) return;

    static GLuint computeProgram = create_compute_program(computeNormalsShader);

    normals_.resize(3*faces_.size());
    GLVector<Point> p(3*faces_.size());

    glUseProgram(computeProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, points_.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, faces_.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, p.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, normals_.gl_id());

    glUniform1ui(0, faces_.size());

    glDispatchCompute((faces_.size() / GroupSize) + 1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    points_ = p;
    faces_.resize(0);
}

}; //namespace display
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os,
                                const rtac::display::GLMesh& mesh)
{
    os << "GLMesh :"
       << "\n- " << mesh.points().size()   << " points"
       << "\n- " << mesh.faces().size()    << " faces"
       << "\n- " << mesh.uvs().size()      << " uvs"
       << "\n- " << mesh.normals().size()  << " faceNormals";
    return os;
}

#endif //_DEF_RTAC_DISPLAY_GL_MESH_H_
