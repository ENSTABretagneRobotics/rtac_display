#include <rtac_display/utils.h>

#include <iomanip>

namespace rtac { namespace display {

GLuint compile_shader(GLenum shaderType, const std::string& source)
{
    GLuint shaderId = glCreateShader(shaderType);
    //check_gl("compile_shader, ShaderId creation failure.");
    GL_CHECK_LAST();

    if(shaderId == 0)
        throw std::runtime_error("could not create shader");
    
    const GLchar* sourceStr = static_cast<const GLchar*>(source.c_str());
    glShaderSource(shaderId, 1, &sourceStr, 0);
    glCompileShader(shaderId);
    //check_gl("Shader compilation failure");
    GL_CHECK_LAST();

    GLint compilationStatus(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compilationStatus);
    if(compilationStatus != GL_TRUE)
    {
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        std::shared_ptr<char> error(new char[errorSize + 1]);
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error.get());
        error.get()[errorSize] = '\0';                  
        glDeleteShader(shaderId);                       
        shaderId = 0;

        std::ostringstream oss;
        oss << "Shader compilation  error :\n";
        oss << std::setfill(' ');
        std::istringstream iss(source);
        std::string line;
        for(int i = 0; std::getline(iss, line); i++) {
            oss << std::setw(4) << i+1 << " " << line << std::endl;
        }
        oss << std::string(error.get()) << std::endl;

        throw std::runtime_error(oss.str());
    }

    return shaderId;
}

GLuint create_render_program(const std::string& vertexShaderSource,
                             const std::string& fragmentShaderSource)
{
    if(vertexShaderSource.size() == 0 || fragmentShaderSource.size() == 0) {
        return 0;
    }
    GLuint vertexShader   = compile_shader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compile_shader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    GLuint programId = glCreateProgram();
    //check_gl("Program creation failure.");
    GL_CHECK_LAST();

    if(programId == 0)
        throw std::runtime_error("Could not create program.");

    glAttachShader(programId, vertexShader);
    glAttachShader(programId, fragmentShader);

    glLinkProgram(programId);

    GLint linkStatus(0);
    glGetProgramiv(programId, GL_LINK_STATUS, &linkStatus);
    if(linkStatus != GL_TRUE)
    {
        std::cout << "Vertex shader :\n" << vertexShaderSource << std::endl;
        std::cout << "Fragment shader :\n" << fragmentShaderSource << std::endl;
        GLint errorSize(0);
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &errorSize);
        std::shared_ptr<char> error(new char[errorSize + 1]);
        glGetProgramInfoLog(programId, errorSize, &errorSize, error.get());
        error.get()[errorSize] = '\0';
        glDeleteProgram(programId);
        programId = 0;

        throw std::runtime_error("Program link error :\n" + std::string(error.get()));
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return programId;
}

GLuint create_compute_program(const std::string& computeShaderSource)
{
    GLuint computeShader = compile_shader(GL_COMPUTE_SHADER, computeShaderSource);

    GLuint programId = glCreateProgram();
    //check_gl("Program creation failure.");
    GL_CHECK_LAST();

    if(programId == 0)
        throw std::runtime_error("Could not create program.");

    glAttachShader(programId, computeShader);

    glLinkProgram(programId);

    GLint linkStatus(0);
    glGetProgramiv(programId, GL_LINK_STATUS, &linkStatus);
    if(linkStatus != GL_TRUE)
    {
        std::cout << "Compute shader :\n" << computeShaderSource << std::endl;
        GLint errorSize(0);
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &errorSize);
        std::shared_ptr<char> error(new char[errorSize + 1]);
        glGetProgramInfoLog(programId, errorSize, &errorSize, error.get());
        error.get()[errorSize] = '\0';
        glDeleteProgram(programId);
        programId = 0;

        throw std::runtime_error("Program link error :\n" + std::string(error.get()));
    }
    glDeleteShader(computeShader);

    return programId;
}

GLuint create_compute_program(const std::vector<std::string>& computeShaderSources)
{
    GLuint programId = glCreateProgram();
    //check_gl("Program creation failure.");
    GL_CHECK_LAST();

    if(programId == 0)
        throw std::runtime_error("Could not create program.");

    for(auto& s : computeShaderSources) {
        glAttachShader(programId, compile_shader(GL_COMPUTE_SHADER, s));
    }

    glLinkProgram(programId);

    GLint linkStatus(0);
    glGetProgramiv(programId, GL_LINK_STATUS, &linkStatus);
    if(linkStatus != GL_TRUE)
    {
        //std::cout << "Compute shader :\n" << computeShaderSource << std::endl;
        for(auto& s : computeShaderSources) {
            std::cout << "Compute shader :\n" << s << std::endl;
        }
        GLint errorSize(0);
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &errorSize);
        std::shared_ptr<char> error(new char[errorSize + 1]);
        glGetProgramInfoLog(programId, errorSize, &errorSize, error.get());
        error.get()[errorSize] = '\0';
        glDeleteProgram(programId);
        programId = 0;

        throw std::runtime_error("Program link error :\n" + std::string(error.get()));
    }
    //glDeleteShader(computeShader);

    return programId;
}
}; //namespace display
}; //namespace rtac
