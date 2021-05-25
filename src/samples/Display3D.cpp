#include <rtac_display/samples/Display3D.h>

namespace rtac { namespace display { namespace samples {

Display3D::Display3D(int width, int height, const std::string& name) :
    view_(PinholeView::New()),
    controls_(OrbitControls::Create(view_, {0,0,0}))
{
    this->add_view(view_);
    this->add_event_handler(controls_);
}

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
