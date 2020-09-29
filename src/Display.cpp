#include <optix_helpers/display/Display.h>

namespace optix_helpers { namespace display {

const Source Display::vertexShader = Source( R"(
#version 430 core

in vec2 point;
out vec2 uv;
uniform mat4 view;

void main()
{
    //gl_Position = vec4(point, 0.0, 1.0);
    gl_Position = view*vec4(point, 0.0, 1.0);
    //uv = 0.5f*(point.xy + 1.0f);
    uv.x = 0.5f*(point.x + 1.0f);
    uv.y = 0.5f*(1.0f - point.y);
}
)", "vertex");

const Source Display::fragmentShader = Source(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;

out vec4 outColor;

void main()
{
    outColor = texture(tex, uv);
}
)", "fragment");

Display::Display(int width, int height, const std::string& title) :
    window_(NULL),
    displayProgram_(0),
    texId_(0),
    imageSize_({0,0})
{
    if(!glfwInit()) {
        throw std::runtime_error("GLFW initialization failure.");
    }
    window_ = Window(glfwCreateWindow(width, height, title.c_str(), NULL, NULL),
                     glfwDestroyWindow); //custom deleter
    if(!window_) {
        throw std::runtime_error("GLFW window creation failure.");
    }
    glfwMakeContextCurrent(window_.get());
    
    // init glew (no gl function availabel if not done)
    GLenum initGlewStatus(glewInit());
    if(initGlewStatus != GLEW_OK)
        std::cout << "Failed to initialize glew" << std::endl;
    //std::cout << glGetString(GL_VERSION) << std::endl;
    //if(GLEW_ARB_compute_shader)
    //    std::cout << "Compute shader ok !" << std::endl;

    glClearColor(0.0,0.0,0.0,1.0);
    //glClearColor(0.7,0.7,0.7,1.0);

    // to measure fps
    glfwSwapInterval(0);

    glViewport(0.0,0.0,width,height);
    displayProgram_ = create_render_program(vertexShader, fragmentShader);
    this->init_texture();
}

void Display::init_texture()
{
    if(!texId_)
        glGenTextures(1, &texId_);

    glBindTexture(GL_TEXTURE_2D, texId_);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Display::terminate()
{
    glDeleteTextures(1, &texId_);
    glDeleteProgram(displayProgram_);
    glfwTerminate();
}

void Display::set_image(int width, int height, const float* data)
{
    imageSize_.width  = width;
    imageSize_.height = height;
    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
        0, GL_RGB, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Display::set_buffer(int width, int height, GLuint bufferId)
{
    imageSize_.width  = width;
    imageSize_.height = height;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferId);
    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
        0, GL_RGB, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

Display::Shape Display::window_shape() const
{
    Shape wSize;
    int width, height;
    glfwGetWindowSize(window_.get(), &width, &height);
    wSize.width  = width;
    wSize.height = height;
    return wSize;
}

int Display::should_close() const
{
    glfwPollEvents();
    return glfwWindowShouldClose(window_.get()) > 0;
}

void Display::wait_for_close() const
{
    using namespace std;
    while(!this->should_close()) {
        std::this_thread::sleep_for(100ms);
    }
}

GLuint Display::create_buffer(size_t size) const
{
    glfwMakeContextCurrent(window_.get());
    GLuint id;
    glGenBuffers(1, &id);
    glBindBuffer(GL_ARRAY_BUFFER, id);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return id;
}

void Display::add_renderer(const Renderer& renderer)
{
    renderers_.push_back(renderer);
}

void Display::draw()
{
    glfwMakeContextCurrent(window_.get());
    Shape wSize = this->window_shape();

    glViewport(0,0,wSize.width,wSize.height);
    glClear(GL_COLOR_BUFFER_BIT);
    
    //this->draw_old();
    
    for(auto renderer : renderers_) {
        if(renderer) {
            renderer->set_screen_size(wSize);
            renderer->draw();
        }
    }

    glfwSwapBuffers(window_.get());
}


void Display::draw_old()
{
    float vertices[] = {-1.0,-1.0,
                         1.0,-1.0,
                         1.0, 1.0,
                        -1.0, 1.0};
                       
    float colors1[] = {1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       1.0, 1.0, 1.0,
                       0.0, 1.0, 0.0};
    unsigned int indexes[] = {0, 1, 2,
                              0, 2, 3};
    
    
    glfwMakeContextCurrent(window_.get());

    float viewMatrix[16] = {1.0,0.0,0.0,0.0,
                            0.0,1.0,0.0,0.0,
                            0.0,0.0,1.0,0.0,
                            0.0,0.0,0.0,1.0};
    Shape wSize;
    int width, height;
    glfwGetWindowSize(window_.get(), &width, &height);
    wSize.width  = width;
    wSize.height = height;

    if(wSize.ratio<float>() > imageSize_.ratio<float>()) {
        viewMatrix[0] = imageSize_.ratio<float>() / wSize.ratio<float>();
    }
    else {
        viewMatrix[5] = wSize.ratio<float>() / imageSize_.ratio<float>();
    }
    glViewport(0,0,wSize.width,wSize.height);

    glClear(GL_COLOR_BUFFER_BIT);
    ////glLineWidth(11);
    ////glColor3f(0.7,0.7,0.0);

    glUseProgram(displayProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, colors1);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(displayProgram_, "view"),
        1, GL_FALSE, viewMatrix);


    glUniform1i(glGetUniformLocation(displayProgram_, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texId_);
    
    //glDrawElements(GL_LINE_STRIP, 3, GL_UNSIGNED_INT, indexes);
    //glDrawElements(GL_LINE_STRIP, 4, GL_UNSIGNED_INT, indexes);
    //glDrawElements(GL_LINE_LOOP, 3, GL_UNSIGNED_INT, indexes);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    //glDrawArrays(GL_TRIANGLES, 0, 3);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);

    glfwSwapBuffers(window_.get());
}

}; //namespace display
}; //namespace optix_helpers


