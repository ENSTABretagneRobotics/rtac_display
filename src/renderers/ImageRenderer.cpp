#include <rtac_display/renderers/ImageRenderer.h>

namespace rtac { namespace display {

/**
 * This vertex shader expects coordinates of the corner of the image, and
 * automatically generates texture coordinates accordingly.
 */
const std::string ImageRenderer::vertexShader = std::string( R"(
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
)");

/**
 * Simply outputs the texture value at given texture coordinates.
 */
const std::string ImageRenderer::fragmentShader = std::string(R"(
#version 430 core

in vec2 uv;
uniform sampler2D tex;

out vec4 outColor;

void main()
{
    outColor = texture(tex, uv);
}
)");

/**
 * Creates a new ImageRenderer object on the heap and outputs a shared_ptr.
 *
 * An OpenGL context must have been created beforehand.
 */
ImageRenderer::Ptr ImageRenderer::New()
{
    return Ptr(new ImageRenderer());
}

/**
 * ImageRenderer does not expects any parameters. It creates its own view.
 *
 * An OpenGL context must have been created before any instantiation.
 */
ImageRenderer::ImageRenderer() :
    Renderer(vertexShader, fragmentShader, ImageView::New()),
    texture_(GLTexture::New()),
    imageView_(std::dynamic_pointer_cast<ImageView>(view_))
{}

GLTexture::Ptr& ImageRenderer::texture()
{
    return texture_;
}

GLTexture::ConstPtr ImageRenderer::texture() const
{
    return texture_;
}

/**
 * Generate screen coordinates of corner of image and displays the image.
 */
void ImageRenderer::draw()
{
    imageView_->set_image_shape(texture_->shape());

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
    glBindTexture(GL_TEXTURE_2D, texture_->gl_id());
    
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indexes);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);

    glUseProgram(0);

    GL_CHECK_LAST();
}

}; //namespace display
}; //namespace rtac

