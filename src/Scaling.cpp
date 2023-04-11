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
