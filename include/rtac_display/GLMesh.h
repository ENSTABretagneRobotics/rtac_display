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
    static const std::string expandVerticesShader;
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

    void compute_normals();
    void expand_vertices();

    static Ptr cube(float scale = 1.0f);
    static Ptr cube_with_uvs(float scale = 1.0f);
    static Ptr from_ply(const std::string& path,
                        bool transposeUVs = false);

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
    static const GLuint computeProgram = create_compute_program(computeNormalsShader);

    this->expand_vertices();

    if(points_.size() % 3 != 0) {
        std::ostringstream oss;
        oss << "GLMesh::compute_normals : number of point after expansion ("
            << points_.size() << ") is not a mutiple of 3. "
            << "Cannot compute normals.";
        std::runtime_error(oss.str());
    }

    normals_.resize(points_.size());

    glUseProgram(computeProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, points_.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, normals_.gl_id());

    glUniform1ui(0, points_.size() / 3);

    glDispatchCompute(((points_.size() / 3) / GroupSize) + 1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);
}

inline void GLMesh::expand_vertices()
{
    if(faces_.size() == 0) return;

    static const GLuint computeProgram = create_compute_program(expandVerticesShader);
    glUseProgram(computeProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, faces_.gl_id());
    glUniform1ui(0, faces_.size());
    
    if(normals_.size() == points_.size()) { 
        GLVector<Point> normals(3*faces_.size());

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, normals_.gl_id());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, normals.gl_id());

        glUniform1ui(1, 3);

        glDispatchCompute((faces_.size() / GroupSize) + 1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        normals_ = std::move(normals);
    }
    else {
        normals_.resize(0);
    }
    
    if(uvs_.size() == points_.size()) { 
        GLVector<UV> uvs(3*faces_.size());

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, uvs_.gl_id());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, uvs.gl_id());

        glUniform1ui(1, 2);

        glDispatchCompute((faces_.size() / GroupSize) + 1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        uvs_ = std::move(uvs);
    }
    else {
        uvs_.resize(0);
    }

    GLVector<Point> points(3*faces_.size());

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, points_.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, points.gl_id());

    glUniform1ui(1, 3);

    glDispatchCompute((faces_.size() / GroupSize) + 1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glUseProgram(0);

    points_ = std::move(points);
    faces_.resize(0);
}

inline GLMesh::Ptr GLMesh::cube(float scale) {
    return Ptr(new GLMesh(BaseMesh::cube(scale)));
}

inline GLMesh::Ptr GLMesh::cube_with_uvs(float scale) {
    Ptr mesh(new GLMesh(BaseMesh::cube(scale)));
    mesh->compute_normals();

    std::vector<UV> uvs(mesh->points().size());

    for(int i = 0; i < 36; i+=6) {
        uvs[i]     = UV({0.0,0.0});
        uvs[i + 1] = UV({1.0,0.0});
        uvs[i + 2] = UV({1.0,1.0});
        uvs[i + 3] = UV({0.0,0.0});
        uvs[i + 4] = UV({1.0,1.0});
        uvs[i + 5] = UV({0.0,1.0});
    }
    mesh->uvs() = uvs;

    return mesh;
}

inline GLMesh::Ptr GLMesh::from_ply(const std::string& path, bool transposeUVs)
{
    std::ifstream f(path, std::ios::binary | std::ios::in);
    if(!f.is_open()) {
        throw std::runtime_error(
            "PointCloud::from_ply : could not open file for reading " + path);
    }
    happly::PLYData data(f);
    
    if(!data.hasElement("vertex")) {
        throw std::runtime_error(
            "Invalid ply file : No vertex defined in \"" + path + "\"");
    }

    auto mesh = Create();
    {
        // Loading vertices
        auto x = data.getElement("vertex").getProperty<float>("x");
        auto y = data.getElement("vertex").getProperty<float>("y");
        auto z = data.getElement("vertex").getProperty<float>("z");

        mesh->points().resize(x.size());
        auto ptr = mesh->points().map();
        Point* data = ptr;
        for(int i = 0; i < x.size(); i++) {
            data[i].x = x[i];
            data[i].y = y[i];
            data[i].z = z[i];
        }
    }

    if(data.hasElement("face")) {
        // Loading faces
        std::vector<std::string> names({"vertex_indices", "vertex_index"});
        std::vector<std::vector<uint32_t>> f;
        for(auto& name : names) {
            try {
                f = data.getElement("face").getListPropertyAnySign<uint32_t>(name);
                break;
            }
            catch(const std::runtime_error& e) {
                // wrong face index name, trying another
            }
        }
        
        mesh->faces().resize(f.size());
        auto ptr = mesh->faces().map();
        Face* data = ptr;
        for(int i = 0; i < f.size(); i++) {
            data[i].x = f[i][0];
            data[i].y = f[i][1];
            data[i].z = f[i][2];
        }
    }

    if(data.hasElement("texCoords")) {
        // Loading texture coordinates
        auto x = data.getElement("texCoords").getProperty<float>("x");
        auto y = data.getElement("texCoords").getProperty<float>("y");

        mesh->uvs().resize(x.size());
        auto ptr = mesh->uvs().map();
        UV* data = ptr;
        if(transposeUVs) {
            for(int i = 0; i < x.size(); i++) {
                data[i].x = y[i];
                data[i].y = x[i];
            }
        }
        else {
            for(int i = 0; i < x.size(); i++) {
                data[i].x = x[i];
                data[i].y = y[i];
            }
        }
    }
    
    return mesh;
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
