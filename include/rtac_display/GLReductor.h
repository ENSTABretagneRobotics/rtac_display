#ifndef _DEF_RTAC_DISPLAY_GL_REDUCTION_H_
#define _DEF_RTAC_DISPLAY_GL_REDUCTION_H_

#include <iostream>
#include <unordered_map>
#include <regex>

#include <rtac_display/utils.h>
#include <rtac_display/GLFormat.h>
#include <rtac_display/GLSLType.h>
#include <rtac_display/GLVector.h>

namespace rtac { namespace display {

/**
 * The purpose of this class is to provide efficient reduction operators for
 * GLVectors (finding extrema, sum, product of data arrays...).
 *
 * This class use compute shaders to compute the reductions.
 */
class GLReductor
{
    public:

    using Ptr      = rtac::Handle<GLReductor>;
    using ConstPtr = rtac::Handle<const GLReductor>;
    
    static const std::string MainShader;
    static const std::string SumOperatorShader;
    static const std::string SubOperatorShader;
    static const std::string MinOperatorShader;
    static const std::string MaxOperatorShader;

    static constexpr const char*  Typename  = "Typename";
    static constexpr unsigned int BlockSize = 256;

    template <typename T>
    static void reduce_in_place(GLVector<T>& input, GLuint reductionProgram);
    template <typename T, typename T2>
    static void reduce(const GLVector<T>& input, GLuint reductionProgram,
                    GLVector<T2>& tmpData);

    protected:
    
    mutable std::unordered_map<std::string,GLuint> programs_;
    mutable GLVector<uint8_t> tmpData_;

    public:

    GLReductor() {}
    ~GLReductor();

    template <typename T>
    T reduce(const GLVector<T>& input, GLuint reductionProgram) const;

    std::string key(const std::string& glslType, const std::string& op) const;
    GLuint  program(const std::string& glslType, const std::string& op) const;
    GLuint add_program(const std::string& glslType, const std::string& op,
                       const std::string& OperatorShader) const;

    GLuint sum_program(const std::string& glslType) const;
    GLuint sub_program(const std::string& glslType) const;
    GLuint min_program(const std::string& glslType) const;
    GLuint max_program(const std::string& glslType) const;

    // Below are helper function. Nothing special.
    template <typename T> GLuint sum_program() const {
        return this->sum_program(GLSLType<T>::value);
    }
    template <typename T> GLuint sub_program() const {
        return this->sub_program(GLSLType<T>::value);
    }
    template <typename T> GLuint min_program() const {
        return this->min_program(GLSLType<T>::value);
    }
    template <typename T> GLuint max_program() const {
        return this->max_program(GLSLType<T>::value);
    }
    
    template <typename T> T sum(const GLVector<T>& input) const {
        return reduce(input, this->sum_program<T>());
    }
    template <typename T> T sub(const GLVector<T>& input) const {
        return reduce(input, this->sub_program<T>());
    }
    template <typename T> T min(const GLVector<T>& input) const {
        return reduce(input, this->min_program<T>());
    }
    template <typename T> T max(const GLVector<T>& input) const {
        return reduce(input, this->max_program<T>());
    }

    template <typename T> void sum_in_place(GLVector<T>& input) const {
        reduce_in_place(input, this->sum_program<T>());
    }
    template <typename T> void sub_in_place(GLVector<T>& input) const {
        reduce_in_place(input, this->sub_program<T>());
    }
    template <typename T> void min_in_place(GLVector<T>& input) const {
        reduce_in_place(input, this->min_program<T>());
    }
    template <typename T> void max_in_place(GLVector<T>& input) const {
        reduce_in_place(input, this->max_program<T>());
    }
};

template <typename T>
T GLReductor::reduce(const GLVector<T>& input, GLuint reductionProgram) const
{
    reduce(input, reductionProgram, tmpData_);
    T res;
    {
        auto p = tmpData_.map();
        res = reinterpret_cast<const T*>(&p[0])[0];
    }
    return res;
}

inline GLuint GLReductor::program(const std::string& glslType, const std::string& op) const
{
    auto it = programs_.find(this->key(glslType, op));
    if(it != programs_.end()) {
        return it->second;
    }
    return 0;
}

inline std::string GLReductor::key(const std::string& glslType, const std::string& op) const
{
    return glslType + "_" + op;
}

template <typename T>
void GLReductor::reduce_in_place(GLVector<T>& input, GLuint reductionProgram)
{
    GLReductor::reduce(input, reductionProgram, input);
}

template <typename T, typename T2>
void GLReductor::reduce(const GLVector<T>& input, GLuint reductionProgram,
                     GLVector<T2>& output)
{
    unsigned int N = input.size();
    unsigned int blockCount = N / (2*BlockSize);

    if(output.size()*sizeof(T2) < blockCount*sizeof(T)) {
        output.resize((blockCount*sizeof(T) + sizeof(T2) - 1) / sizeof(T2));
    }

    glUseProgram(reductionProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input.gl_id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output.gl_id());

    if(N > 0) {
        glUniform1ui(0, N);
        if(blockCount == 0)
            glDispatchCompute(1,1,1);
        else
            glDispatchCompute(blockCount,1,1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        N = blockCount;
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, output.gl_id());
    
    while(N > 0) {
        blockCount = N / (2*BlockSize);

        glUniform1ui(0, N);
        if(blockCount == 0)
            glDispatchCompute(1,1,1);
        else
            glDispatchCompute(blockCount,1,1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        N = blockCount;
    }

    output.unbind(GL_SHADER_STORAGE_BUFFER);
    input.unbind(GL_SHADER_STORAGE_BUFFER);
    glUseProgram(0);
    GL_CHECK_LAST();
}

inline GLuint GLReductor::sum_program(const std::string& glslType) const
{
    auto program = this->program(glslType, "sum");
    if(program)
        return program;
    return this->add_program(glslType, "sum", SumOperatorShader);
}

inline GLuint GLReductor::sub_program(const std::string& glslType) const
{
    auto program = this->program(glslType, "sub");
    if(program)
        return program;
    return this->add_program(glslType, "sub", SubOperatorShader);
}

inline GLuint GLReductor::min_program(const std::string& glslType) const
{
    auto program = this->program(glslType, "min");
    if(program)
        return program;
    return this->add_program(glslType, "min", MinOperatorShader);
}

inline GLuint GLReductor::max_program(const std::string& glslType) const
{
    auto program = this->program(glslType, "max");
    if(program)
        return program;
    return this->add_program(glslType, "max", MaxOperatorShader);
}

}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_GL_REDUCTION_H_
