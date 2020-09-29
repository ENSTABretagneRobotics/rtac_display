#include <optix_helpers/display/ImageRenderer.h>

namespace optix_helpers { namespace display {

const Source ImageRendererObj::vertexShader = Source( R"(
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

const Source ImageRendererObj::fragmentShader = Source(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;

out vec4 outColor;

void main()
{
    outColor = texture(tex, uv);
}
)", "fragment");

ImageRendererObj::ImageRendererObj() :
    RendererObj(vertexShader, fragmentShader, ImageView::New()),
    texId_(0),
    imageView_(std::dynamic_pointer_cast<ImageViewObj>(view_.ptr()))
{
    this->init_texture();
}

ImageRendererObj::~ImageRendererObj()
{
    glDeleteTextures(1, &texId_);
}

void ImageRendererObj::init_texture()
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

void ImageRendererObj::set_image(const Shape& imageSize, GLuint bufferId)
{
    // only for RGB data
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferId);
    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageSize.width, imageSize.height,
        0, GL_RGB, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    imageView_->set_image_shape(imageSize);
}

void ImageRendererObj::set_image(const RenderBufferGL& buffer)
{
    this->set_image(buffer->shape(), buffer->gl_id());
}

void ImageRendererObj::draw()
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
    


    glUseProgram(renderProgram_);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, colors1);
    glEnableVertexAttribArray(1);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, imageView_->view_matrix().data());


    glUniform1i(glGetUniformLocation(renderProgram_, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texId_);
    
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);
}

}; //namespace display
}; //namespace optix_helpers

