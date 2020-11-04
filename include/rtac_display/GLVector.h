#ifndef _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
#define _DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

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

    protected:

    GLuint bufferId_;
    size_t size_;

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

    //pointer       data();
    //const_pointer data() const;

    //iterator begin();
    //iterator end();
    //const_iterator begin() const;
    //const_iterator end() const;

    // not really const but no other way with OpenGL interface
    GLuint gl_id() const;

    // work needed here
    void bind(GLenum target = GL_ARRAY_BUFFER) const; 
    void unbind(GLenum target = GL_ARRAY_BUFFER) const;
};

// implementation
template <typename T>
GLVector<T>::GLVector() :
    bufferId_(0),
    size_(0)
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

//template <typename T> typename GLVector<T>::
//pointer GLVector<T>::data()
//{
//    return data_;
//}
//
//template <typename T> typename GLVector<T>::
//const_pointer GLVector<T>::data() const
//{
//    return data_;
//}
//
//template <typename T> typename GLVector<T>::
//iterator GLVector<T>::begin()
//{
//    return data_;
//}
//
//template <typename T> typename GLVector<T>::
//iterator GLVector<T>::end()
//{
//    return data_ + size_;
//}
//
//template <typename T> typename GLVector<T>::
//const_iterator GLVector<T>::begin() const
//{
//    return data_;
//}
//
//template <typename T> typename GLVector<T>::
//const_iterator GLVector<T>::end() const
//{
//    return data_ + size_;
//}

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

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_BASE_DISPLAY_GL_VECTOR_H_
