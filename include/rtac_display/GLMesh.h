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

    protected:
    
    GLVector<Point>  points_;
    GLVector<Face>   faces_;
    GLVector<UV>     uvs_;
    GLVector<Normal> faceNormals_;
    GLVector<Normal> pointNormals_;

    public:

    static Ptr Create() { return Ptr(new GLMesh()); }

    GLMesh() {}
    GLMesh(GLMesh&& other);
    GLMesh& operator=(GLMesh&& other);

    template <template <typename> class VectorT>
    GLMesh(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other);
    template <template <typename> class VectorT>
    GLMesh& operator=(const types::Mesh<GLMesh::Point,GLMesh::Face,VectorT>& other);

    GLVector<Point>&  points()        { return points_; }
    GLVector<Face>&   faces()         { return faces_; }
    GLVector<UV>&     uvs()           { return uvs_; }
    GLVector<Normal>& face_normals()  { return faceNormals_; }
    GLVector<Normal>& point_normals() { return pointNormals_; }

    const GLVector<Point>&  points()        const { return points_; }
    const GLVector<Face>&   faces()         const { return faces_; }
    const GLVector<UV>&     uvs()           const { return uvs_; }
    const GLVector<Normal>& face_normals()  const { return faceNormals_; }
    const GLVector<Normal>& point_normals() const { return pointNormals_; }

    static GLMesh::Ptr cube(float scale = 1.0f) {
        return Ptr(new GLMesh(BaseMesh::cube(scale)));
    }
};

inline GLMesh::GLMesh(GLMesh&& other) :
    GLMesh()
{
    *this = std::move(other);
}

inline GLMesh& GLMesh::operator=(GLMesh&& other)
{
    points_       = std::move(other.points_);
    faces_        = std::move(other.faces_);
    uvs_          = std::move(other.uvs_);
    faceNormals_  = std::move(other.faceNormals_);
    pointNormals_ = std::move(other.pointNormals_);

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
    faceNormals_.resize(0);
    pointNormals_.resize(0);

    return *this;
}

}; //namespace display
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os,
                                const rtac::display::GLMesh& mesh)
{
    os << "GLMesh :"
       << "\n- " << mesh.points().size()       << " points"
       << "\n- " << mesh.faces().size()        << " faces"
       << "\n- " << mesh.uvs().size()          << " uvs"
       << "\n- " << mesh.face_normals().size()  << " faceNormals"
       << "\n- " << mesh.point_normals().size() << " pointNormals";
    return os;
}

#endif //_DEF_RTAC_DISPLAY_GL_MESH_H_
