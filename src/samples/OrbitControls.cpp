#include <rtac_display/samples/OrbitControls.h>

namespace rtac { namespace display { namespace samples {

using namespace rtac::indexing;

OrbitControls::OrbitControls(const View3D::Ptr& view,
                             const Vec3& target,
                             const Vec3& up) :
    EventHandler(false, true, true, true),
    view_(view),
    viewFrame_(Mat3::Identity()),
    orientationLocked_(false),
    targetLocked_(false),
    target_(target),
    angleSensitivity_(0.005),
    zoomSensitivity_(1.1)
{
    viewFrame_(all,2) = up.normalized();
    viewFrame_(all,0) = geometry::find_orthogonal(up).normalized();
    viewFrame_(all,1) = up.cross(viewFrame_(all,0));

    auto p = view_->pose().translation() - target_;
    rho_   = p.norm();
}

OrbitControls::Ptr OrbitControls::Create(const View3D::Ptr& view,
                                         const Vec3& target,
                                         const Vec3& up)
{
    return Ptr(new OrbitControls(view, target, up));
}

void OrbitControls::look_at(const Vec3& target, const Vec3& position)
{
    target_ = target;
    this->view_->look_at(target_, position, viewFrame_(all,2));
    this->update_parameters_from_view();
}

void OrbitControls::update_parameters_from_view()
{
    auto p = view_->pose().translation() - target_;

    rho_   = p.norm();
    phi_   = std::asin(p.normalized().dot(viewFrame_(all,2)));
    theta_ = std::atan2(p.dot(viewFrame_(all,1)), p.dot(viewFrame_(all,0)));
}

void OrbitControls::update_view_from_parameters()
{
    Vec3 p = rho_*viewFrame_*Vec3({std::cos(theta_)*std::cos(phi_),
                                   std::sin(theta_)*std::cos(phi_),
                                   std::sin(phi_)}) + target_;

    view_->look_at(target_, p, viewFrame_(all,2));
}

void OrbitControls::mouse_position_callback(double x, double y)
{
    if(!window_) {
        std::cerr << "Warning : window is null but callback was called." << std::endl;
        return;
    }

    MousePosition dmouse = MousePosition({x,y}) - lastMouse_;
    lastMouse_ = MousePosition({x,y});

    if(orientationLocked_) {
        this->update_orientation(dmouse);
    }
    else if(targetLocked_) {
        this->update_target(dmouse);
    }
}

void OrbitControls::update_orientation(const MousePosition& deltaMouse)
{
    theta_ -= angleSensitivity_*deltaMouse[0];
    phi_   += angleSensitivity_*deltaMouse[1];
    phi_    = std::max(std::min(phi_, 1.5f), -1.5f);
    this->update_view_from_parameters();
}

void OrbitControls::update_target(const MousePosition& deltaMouse)
{
    Vec3 dx = -view_->raw_view_matrix()(seqN(0,3), 0);
    Vec3 dy =  view_->raw_view_matrix()(seqN(0,3), 1);
    Vec3 dz =  view_->raw_view_matrix()(seqN(0,3), 2);

    Vec3 dp = (deltaMouse[0]*dx + deltaMouse[1]*dy) * rho_ / view_->screen_size().height;
    target_ += dp;
    view_->look_at(target_, view_->raw_view_matrix()(seqN(0,3),3) + dp, viewFrame_(all,2));
}

void OrbitControls::mouse_button_callback(int button, int action, int modes)
{
    if(!window_) {
        std::cerr << "Warning : window is null but callback was called." << std::endl;
        return;
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT) {
        if(action == GLFW_PRESS) {
            glfwSetInputMode(window_.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            this->update_parameters_from_view();    
            glfwGetCursorPos(window_.get(), &lastMouse_[0], &lastMouse_[1]);
            if(glfwGetKey(window_.get(), GLFW_KEY_LEFT_SHIFT))
                targetLocked_ = true;
            else
                orientationLocked_ = true;
        }
        else {
            glfwSetInputMode(window_.get(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            orientationLocked_ = false;
            targetLocked_ = false;
        }
    }
}

void OrbitControls::scroll_callback(double x, double y)
{
    this->update_parameters_from_view();
    if(y > 0)
        rho_ /= std::abs(y)*zoomSensitivity_;
    else
        rho_ *= std::abs(y)*zoomSensitivity_;
    this->update_view_from_parameters();
}

}; //namespace samples
}; //namespace display
}; //namespace rtac
