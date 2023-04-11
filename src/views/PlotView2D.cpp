#include <rtac_display/views/PlotView2D.h>

namespace rtac { namespace display {

PlotView2D::PlotView2D(const Bounds<float>& xRange, const Bounds<float>& yRange) :
    View(),
    scaling_(xRange, yRange),
    equalAspect_(false)
{}

PlotView2D::Ptr PlotView2D::Create(const Bounds<float>& xRange,
                                   const Bounds<float>& yRange)
{
    return Ptr(new PlotView2D(xRange, yRange));
}

PlotView2D::Ptr PlotView2D::Create(const PlotData2D::ConstPtr& plotData)
{
    return Create(plotData->x_range(), plotData->y_range());
}

void PlotView2D::update_projection()
{
    Mat4 proj = Mat4::Identity();
    if(!equalAspect_) {
        proj(0,0) = 2.0f / this->x_range().length();
        proj(1,1) = 2.0f / this->y_range().length();
    }
    else {
        auto screenAspect = this->screen_size().ratio<float>();
        auto dataAspect   = this->x_range().length()
                          / this->y_range().length();
        if(dataAspect > screenAspect) {
            proj(0,0) = 2.0f / this->x_range().length();
            proj(1,1) = 2.0f * screenAspect / this->x_range().length();
        }
        else {
            proj(0,0) = 2.0f / (screenAspect * this->y_range().length());
            proj(1,1) = 2.0f / this->y_range().length();
        }
    }

    proj(0,3) = -proj(0,0) * this->x_range().center();
    proj(1,3) = -proj(1,1) * this->y_range().center();
    this->projectionMatrix_ = proj;
}


void PlotView2D::update_ranges(const Bounds<float>& xRange, const Bounds<float>& yRange)
{
    scaling_.update(xRange, yRange);
}

void PlotView2D::update_ranges(const PlotData2D::ConstPtr& plotData)
{
    this->update_ranges(plotData->x_range(), plotData->y_range());
}

} //namespace display
} //namespace rtac

