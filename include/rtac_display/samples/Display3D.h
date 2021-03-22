#ifndef _DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_
#define _DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_

#include <rtac_display/Display.h>
#include <rtac_display/views/PinholeView.h>
#include <rtac_display/samples/OrbitControls.h>

namespace rtac { namespace display { namespace samples {

class Display3D : public Display
{
    protected:

    PinholeView::Ptr   view_;
    OrbitControls::Ptr controls_;

    public:

    Display3D(int width = 1280, int height = 960,
              const std::string& name = "display_3d");

    PinholeView::Ptr view();
    OrbitControls::Ptr controls();
    PinholeView::ConstPtr view() const;
    OrbitControls::ConstPtr controls() const;
};

}; //namespace samples
}; //namespace display
}; //namespace rtac

#endif //_DEF_RTAC_DISPLAY_SAMPLES_DISPLAY_3D_H_
