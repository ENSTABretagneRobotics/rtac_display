#ifndef _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
#define _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <rtac_base/types/MappedPointer.h>
#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_display/utils.h>
#include <rtac_display/cuda/utils.h>

namespace rtac { namespace display {

template <typename T>
class GLVector
{
    public:

    using value_type      = T;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    using MappedPointer      = rtac::types::MappedPointer<GLVector<T>>;
    using ConstMappedPointer = rtac::types::MappedPointer<const GLVector<T>>;

    protected:

    GLuint bufferId_;
    size_t size_;
    mutable T*     mappedPtr_;

    void allocate(size_t size);
    void free();

    public:

    GLVector();
    GLVector(size_t size);
    GLVector(const GLVector<T>& other);
    GLVector(const rtac::cuda::DeviceVector<T>& other);
    GLVector(const rtac::cuda::HostVector<T>& other);
    GLVector(const std::vector<T>& other);
    ~GLVector();
    
    GLVector& operator=(const GLVector<T>& other);
    GLVector& operator=(const rtac::cuda::DeviceVector<T>& other);
    GLVector& operator=(const rtac::cuda::HostVector<T>& other);
    GLVector& operator=(const std::vector<T>& other);

    rtac::cuda::DeviceVector<T>& copy_to_cuda(rtac::cuda::DeviceVector<T>& other) const;
    rtac::cuda::DeviceVector<T>  copy_to_cuda() const;

    void resize(size_t size);
    size_t size() const;
    size_t capacity() const;

    // not really const but no other way with OpenGL interface
    GLuint gl_id() const;

    // work needed here
    void bind(GLenum target = GL_ARRAY_BUFFER) const; 
    void unbind(GLenum target = GL_ARRAY_BUFFER) const;

    // Mapping Functions
    protected:

    T*       do_map();
    T*       do_map_write_only();
    const T* do_map() const;

    public:

    MappedPointer      map(bool writeOnly = false);
    ConstMappedPointer map() const;
    void               unmap() const;
};

// implementation
template <typename T>
GLVector<T>::GLVector() :
    bufferId_(0),
    size_(0),
    mappedPtr_(nullptr)
{}

template <typename T>
GLVector<T>::GLVector(size_t size) :
    GLVector()
{
    this->resize(size);
}

template <typename T>
GLVector<T>::GLVector(const GLVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::GLVector(const rtac::cuda::DeviceVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::GLVector(const rtac::cuda::HostVector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::GLVector(const std::vector<T>& other) :
    GLVector(other.size())
{
    *this = other;
}

template <typename T>
GLVector<T>::~GLVector()
{
    this->free();
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const GLVector<T>& other)
{
    this->resize(other.size());

    other.bind(GL_COPY_READ_BUFFER);
    this->bind(GL_COPY_WRITE_BUFFER);

    glCopyBufferSubData(GL_COPY_READ_BUFFER,
                        GL_COPY_WRITE_BUFFER,
                        0, 0, this->size()*sizeof(T));

    this->unbind(GL_COPY_WRITE_BUFFER);
    other.unbind(GL_COPY_READ_BUFFER);

    return *this;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::DeviceVector<T>& other)
{
    this->resize(other.size());
    cuda::copy_to_gl(this->gl_id(), other.data(), this->size()*sizeof(T));
    return *this;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const rtac::cuda::HostVector<T>& other)
{
    this->resize(other.size());
    this->bind(GL_ARRAY_BUFFER);
    
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), other.data());

    this->unbind(GL_ARRAY_BUFFER);
    return *this;
}

template <typename T>
GLVector<T>& GLVector<T>::operator=(const std::vector<T>& other)
{
    this->resize(other.size());
    this->bind(GL_ARRAY_BUFFER);
    
    glBufferSubData(GL_ARRAY_BUFFER, 0, this->size()*sizeof(T), other.data());

    this->unbind(GL_ARRAY_BUFFER);
    return *this;
}

template <typename T>
rtac::cuda::DeviceVector<T>& GLVector<T>::copy_to_cuda(rtac::cuda::DeviceVector<T>& other) const
{
    other.resize(this->size());
    cuda::copy_from_gl(other.data(), this->gl_id(), this->size()*sizeof(T));
    return other;
}

template <typename T>
rtac::cuda::DeviceVector<T> GLVector<T>::copy_to_cuda() const
{
    rtac::cuda::DeviceVector<T> res(this->size());
    return this->copy_to_cuda(res);
}

template <typename T>
void GLVector<T>::allocate(size_t size)
{
    if(!bufferId_)
        glGenBuffers(1, &bufferId_);
    this->bind();
    glBufferData(GL_ARRAY_BUFFER, size*sizeof(T), NULL, GL_STATIC_DRAW);
    this->unbind();
}

template <typename T>
void GLVector<T>::free()
{
    if(bufferId_)
        glDeleteBuffers(1, &bufferId_);
    bufferId_ = 0;
    size_     = 0;
}

template <typename T>
void GLVector<T>::resize(size_t size)
{
    if(this->capacity() < size)
        this->allocate(size);
    size_ = size;
}

template <typename T>
size_t GLVector<T>::size() const
{
    return size_;
}

template <typename T>
size_t GLVector<T>::capacity() const
{
    if(!bufferId_)
        return 0;
    GLint capa;
    this->bind();
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &capa);
    this->unbind();
    return capa / sizeof(T);
}

template <typename T>
GLuint GLVector<T>::gl_id() const
{
    return bufferId_;
}

template <typename T>
void GLVector<T>::bind(GLenum target) const
{
    glBindBuffer(target, this->gl_id());
    check_gl("GLVector::bind : could not bind buffer.");
}

template <typename T>
void GLVector<T>::unbind(GLenum target) const
{
    glBindBuffer(target, 0);
    check_gl("GLVector::unbind : could not unbind buffer (invalid target).");
}

// Mapping functions
template <typename T>
T* GLVector<T>::do_map()
{
    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}

template <typename T>
T* GLVector<T>::do_map_write_only()
{
    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}


template <typename T>
const T* GLVector<T>::do_map() const
{
    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for mapping.");
    mappedPtr_ = static_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    check_gl("GLVector : could not map.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return mappedPtr_;
}

template <typename T>
typename GLVector<T>::MappedPointer GLVector<T>::map(bool writeOnly)
{
    if(mappedPtr_)
        throw std::runtime_error("GLVector already mapped. Cannot map a second time.");

    if(writeOnly) {
        return MappedPointer(this,
                             &GLVector<T>::do_map_write_only,
                             &GLVector<T>::unmap);
    }
    else {
        return MappedPointer(this,
                             &GLVector<T>::do_map,
                             &GLVector<T>::unmap);
    }
}

template <typename T>
typename GLVector<T>::ConstMappedPointer GLVector<T>::map() const
{
    if(mappedPtr_)
        throw std::runtime_error("GLVector already mapped. Cannot map a second time.");

    return ConstMappedPointer(this,
                              &GLVector<T>::do_map,
                              &GLVector<T>::unmap);
}

template <typename T>
void GLVector<T>::unmap() const
{
    glBindBuffer(GL_ARRAY_BUFFER, bufferId_);
    check_gl("GLVector : could not bind buffer for unmapping.");
    glUnmapBuffer(GL_ARRAY_BUFFER);
    check_gl("GLVector : could not  unmap.");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    mappedPtr_ = nullptr;
}

}; //namespace display
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::display::GLVector<T>& v)
{
    os << "(";
    auto data = v.map();
    if(v.size() <= 16) {
        os << data[0];
        for(int i = 1; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    else {
        for(int i = 0; i < 3; i++) {
            os << data[i] << " ";
        }
        os << "...";
        for(int i = v.size() - 3; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    os << ")";
    return os;
}

#endif //_DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
