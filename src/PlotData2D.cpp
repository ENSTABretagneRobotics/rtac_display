#include <rtac_display/PlotData2D.h>

namespace rtac { namespace display {

PlotData2D::PlotData2D() : 
    xRange_(0,1),
    yRange_(0,1)
{}

void PlotData2D::update_ranges()
{
    if(y_.size() == 0)
        return;
    if(x_.size() == 0) {
        xRange_.lower = 0;
        xRange_.upper = y_.size();
    }
    else {
        xRange_.lower = reductor_.min(x_);
        xRange_.upper = reductor_.max(x_);
    }

    yRange_.lower = reductor_.min(y_);
    yRange_.upper = reductor_.max(y_);
}

} //namespace display
} //namespace rtac


