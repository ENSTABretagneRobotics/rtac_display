#include <rtac_display/Scaling.h>

namespace rtac { namespace display {

ScalingMode::ScalingMode(bool autoscale,
                         bool memory,
                         bool useOrigin,
                         bool useLimits) :
    autoscale_(autoscale),
    memory_(memory),
    useOrigin_(useOrigin),
    useLimits_(useLimits)
{}


Scaling1D::Scaling1D(const ScalingMode& mode, const Bounds<float>& range) :
    mode_(mode),
    range_(range),
    origin_(range.center()),
    limits_(range)
{}

Scaling1D::Scaling1D(const Bounds<float>& range) :
    Scaling1D(ScalingMode(), range)
{}

void Scaling1D::enable_origin()
{
    this->enable_origin(origin_);
}

void Scaling1D::enable_origin(float origin)
{
    this->set_origin(origin, true);
}

void Scaling1D::set_origin(float origin, bool enableOrigin)
{
    origin_ = origin;
    if(enableOrigin)
        mode_.useOrigin_ = true;
    this->update(range_);
}

void Scaling1D::enable_limits()
{
    this->enable_limits(limits_);
}

void Scaling1D::enable_limits(const Bounds<float>& limits)
{
    this->set_limits(limits, true);
}

void Scaling1D::set_limits(const Bounds<float>& limits, bool enableLimits)
{
    limits_ = limits;
    if(enableLimits) {
        mode_.useLimits_ = true;
        range_.intersect_with(limits_);
    }
    this->update(range_);
}

void Scaling1D::set_range(const Bounds<float>& range, bool disableAutoscale)
{
    range_ = range;
    if(disableAutoscale)
        mode_.autoscale_ = false;
}


/**
 * Updates internal range_. Return true if range_ changed.
 */
bool Scaling1D::update(Bounds<float> range)
{
    if(!mode_.autoscale_)
        return false;
    if(mode_.useLimits_) {
        range.intersect_with(limits_);
    }
    if(mode_.useOrigin_) {
        range.update(origin_); // this ensures origin is in range_
    }
    if(mode_.memory_) { // this ensure old range_ is contained within new range
        if(range_.contains(range)) {
            return false;
        }
        range_.update(range);
    }
    else {
        if(range_ == range) {
            return false;
        }
        range_ = range;
    }
    return true;
}

// Scaling2D ////////////////////////////////////

Scaling2D::Scaling2D(const ScalingMode& mode,
                     const Bounds<float>& xRange,
                     const Bounds<float>& yRange) :
    xScaling_(mode, xRange),
    yScaling_(mode, yRange)
{}

Scaling2D::Scaling2D(const Bounds<float>& xRange, const Bounds<float>& yRange) :
    Scaling2D(ScalingMode(), xRange, yRange)
{}

Point2<float> Scaling2D::origin() const
{
    return Point2<float>({xScaling_.origin(), yScaling_.origin()});
}

void Scaling2D::enable_autoscale()
{
    xScaling_.enable_autoscale();
    yScaling_.enable_autoscale();
}

void Scaling2D::disable_autoscale()
{
    xScaling_.disable_autoscale();
    yScaling_.disable_autoscale();
}

void Scaling2D::enable_memory()
{
    xScaling_.enable_memory();
    yScaling_.enable_memory();
}

void Scaling2D::disable_memory()
{
    xScaling_.disable_memory();
    yScaling_.disable_memory();
}

void Scaling2D::enable_origin()
{
    xScaling_.enable_origin();
    yScaling_.enable_origin();
}

void Scaling2D::enable_origin(Point2<float> origin)
{
    xScaling_.enable_origin(origin.x);
    yScaling_.enable_origin(origin.y);
}

void Scaling2D::set_origin(Point2<float> origin, bool enableOrigin)
{
    xScaling_.set_origin(origin.x, enableOrigin);
    yScaling_.set_origin(origin.y, enableOrigin);
}

void Scaling2D::disable_origin()
{
    xScaling_.disable_origin();
    yScaling_.disable_origin();
}

void Scaling2D::enable_limits()
{
    xScaling_.enable_limits();
    yScaling_.enable_limits();
}

void Scaling2D::enable_limits(const Bounds<float>& xLimits, const Bounds<float>& yLimits)
{
    xScaling_.enable_limits(xLimits);
    yScaling_.enable_limits(yLimits);
}

void Scaling2D::set_limits(const Bounds<float>& xLimits, const Bounds<float>& yLimits,
                           bool enableLimits)
{
    xScaling_.set_limits(xLimits, enableLimits);
    yScaling_.set_limits(yLimits, enableLimits);
}

void Scaling2D::disable_limits()
{
    xScaling_.disable_limits();
    yScaling_.disable_limits();
}

void Scaling2D::set_range(const Bounds<float>& xRange, const Bounds<float>& yRange,
                          bool disableAutoscale)
{
    xScaling_.set_range(xRange, disableAutoscale);
    yScaling_.set_range(yRange, disableAutoscale);
}

bool Scaling2D::update(const Bounds<float>& xRange, const Bounds<float>& yRange)
{
    return xScaling_.update(xRange) || yScaling_.update(yRange);
}

} //namespace display
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::display::ScalingMode& mode)
{
    os << "autoscale:"    << mode.autoscale_
       << ", memory:"     << mode.memory_
       << ", use_origin:" << mode.useOrigin_
       << ", use_limits:" << mode.useLimits_;
    return os;
}

std::ostream& operator<<(std::ostream& os, const rtac::display::Scaling1D& scaling)
{
    os << "Scaling1D (" << scaling.mode() << ')'
       << "\n- range  : " << scaling.range()
       << "\n- origin : " << scaling.origin()
       << "\n- limits : " << scaling.limits();
    return os;
}
