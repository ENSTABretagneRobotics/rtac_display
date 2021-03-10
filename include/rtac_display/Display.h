#ifndef _DEF_RTAC_DISPLAY_H_
#define _DEF_RTAC_DISPLAY_H_

#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

//#include <glm/glm.hpp>
//#include <glm/gtx/transform.hpp>
//#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include <rtac_base/time.h>

#include <rtac_display/utils.h>
#include <rtac_display/views/View.h>
#include <rtac_display/renderers/Renderer.h>

namespace rtac { namespace display {

class Display
{
    public:

    using Window    = std::shared_ptr<GLFWwindow>;
    using Shape     = View::Shape;
    using Views     = std::vector<View::Ptr>;
    using Renderers = std::vector<Renderer::Ptr>;

    protected:
    
    Window    window_;
    Views     views_;
    Renderers renderers_;
    
    rtac::time::FrameCounter frameCounter_;
    bool frameCounterEnabled_;

    public:

    Display(int width = 800, int height = 600,
            const std::string& title = "optix render");
    void terminate();

    Shape window_shape() const;
    int should_close() const;
    int is_drawing();
    void wait_for_close() const;
    
    void add_view(const View::Ptr& view);
    void add_renderer(const Renderer::Ptr& renderer);
    virtual void draw();

    void enable_frame_counter();
    void disable_frame_counter();
};

}; //namespace display
}; //namespace rtac


#endif //_DEF_OPTIX_HELPERS_DISPLAY_H_
