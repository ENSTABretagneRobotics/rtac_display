#include <rtac_display/samples/Display3D.h>

namespace rtac { namespace display { namespace samples {

Display3D::Display3D(int width, int height, const std::string& name,
                     const Display::Context::Ptr& sharedContext) :
    Display(width,height,name,sharedContext),
    view_(PinholeView::Create()),
    controls_(OrbitControls::Create(view_, {0,0,0}))
{
    this->add_view(view_);
    this->add_event_handler(controls_);
    this->view()->look_at({0,0,0},{5,4,3});

    glEnable(GL_DEPTH_TEST);
}

Display3D::Display3D(const Display::Context::Ptr& sharedContext) :
    Display3D(800, 600, "rtac_display", sharedContext)
{}

PinholeView::Ptr Display3D::view()
{
    return view_;
}

OrbitControls::Ptr Display3D::controls()
{
    return controls_;
}

PinholeView::ConstPtr Display3D::view() const
{
    return view_;
}

OrbitControls::ConstPtr Display3D::controls() const
{
    return controls_;
}


}; //namespace samples
}; //namespace display
}; //namespace rtac

