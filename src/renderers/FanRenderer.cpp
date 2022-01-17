#include <rtac_display/renderers/FanRenderer.h>

namespace rtac { namespace display {

const std::string& FanRenderer::vertexShader = std::string(R"(
#version 430 core

in  vec4 point;
out vec2 xyPos;

uniform mat4 view;

void main()
{
    gl_Position = view*point;
    xyPos = point.xy;
}
)");

const std::string& FanRenderer::fragmentShader = std::string(R"(
#version 430 core

in      vec2 xyPos;
uniform vec2 angleBounds;
uniform vec2 rangeBounds;

//uniform sampler2D fanData;
uniform sampler2D colormap;

out vec4 outColor;

bool angle_inside(float angle, vec2 bounds) {
    if(bounds.x < bounds.y) {
        return bounds.x <= angle && angle <= bounds.y;
    }
    else {
        return !(bounds.y < angle && angle < bounds.x);
    }
}

void main()
{
    float r     = length(xyPos);
    float theta = atan(xyPos.y, xyPos.x);
    if(r >= rangeBounds.x && r <= rangeBounds.y && angle_inside(theta, angleBounds))
    {
        outColor = texture(colormap, vec2(1.0f,0.0f));
    }
    else
    {
        outColor = texture(colormap, vec2(0.0f,0.0f));
    }
}
)");

FanRenderer::FanRenderer(const GLContext::Ptr& context) :
    Renderer(context, vertexShader, fragmentShader),
    data_(GLTexture::New()),
    angle_({-M_PI, M_PI}),
    range_({0.0f,1.0f}),
    corners_(6),
    colormap_(colormap::Viridis())
{
    this->set_geometry(angle_, range_);
}

FanRenderer::Ptr FanRenderer::Create(const GLContext::Ptr& context)
{
    return Ptr(new FanRenderer(context));
}

void FanRenderer::set_geometry_degrees(const Interval& angle, const Interval& range,
                                       Direction dir)
{
    std::cout << angle  << std::endl;
    this->set_geometry({(float)(angle.min * M_PI / 180.0f),
                        (float)(angle.max * M_PI / 180.0f)},
                       range, dir);
}

void FanRenderer::set_geometry(Interval angle, const Interval& range, Direction dir)
{
    using Point2 = rtac::types::Point2<float>;

    std::cout << angle  << std::endl;
    switch(dir) {
        default:break;
        case Direction::Bottom:
            angle.min -= 0.5f*M_PI;
            angle.max -= 0.5f*M_PI;
            break;
        case Direction::Top:
            angle.min += 0.5f*M_PI;
            angle.max += 0.5f*M_PI;
            break;
        case Direction::Right:
            angle.min += M_PI;
            angle.max += M_PI;
            break;
    };
    
    // finding extermas of the fan display area.
    std::vector<Point2> poi;
    poi.push_back({range.min*std::cos(angle.min), range.min*std::sin(angle.min)});
    poi.push_back({range.min*std::cos(angle.max), range.min*std::sin(angle.max)});
    poi.push_back({range.max*std::cos(angle.min), range.max*std::sin(angle.min)});
    poi.push_back({range.max*std::cos(angle.max), range.max*std::sin(angle.max)});
    if(angle.is_inside(0.0f)) {
        poi.push_back({range.min, 0.0});
        poi.push_back({range.max, 0.0});
    }
    if(angle.is_inside(M_PI)) {
        poi.push_back({-range.min, 0.0});
        poi.push_back({-range.max, 0.0});
    }
    if(angle.is_inside(0.5f*M_PI)) {
        poi.push_back({0.0, range.min});
        poi.push_back({0.0, range.max});
    }
    if(angle.is_inside(1.5f*M_PI) || angle.is_inside(-0.5f*M_PI)) {
        poi.push_back({0.0, -range.min});
        poi.push_back({0.0, -range.max});
    }

    bounds_.left   = poi[0].x;
    bounds_.right  = poi[0].y;
    bounds_.top    = poi[0].y;
    bounds_.bottom = poi[0].y;
    for(auto p : poi) {
        bounds_.left   = std::min(bounds_.left,   p.x);
        bounds_.right  = std::max(bounds_.right,  p.x);
        bounds_.top    = std::max(bounds_.top,    p.y);
        bounds_.bottom = std::min(bounds_.bottom, p.y);
    }

    while(angle.min >  M_PI) angle.min -= 2*M_PI;
    while(angle.min < -M_PI) angle.min += 2*M_PI;
    while(angle.max >  M_PI) angle.max -= 2*M_PI;
    while(angle.max < -M_PI) angle.max += 2*M_PI;
    angle_ = angle;
    range_ = range;
    
    auto p = corners_.map();
    p[0] = Point4({bounds_.left,  bounds_.bottom, 0.0f, 1.0f});
    p[1] = Point4({bounds_.right, bounds_.bottom, 0.0f, 1.0f});
    p[2] = Point4({bounds_.right, bounds_.top,    0.0f, 1.0f});
    p[3] = Point4({bounds_.left,  bounds_.bottom, 0.0f, 1.0f});
    p[4] = Point4({bounds_.right, bounds_.top,    0.0f, 1.0f});
    p[5] = Point4({bounds_.left,  bounds_.top,    0.0f, 1.0f});
}

FanRenderer::Mat4 FanRenderer::compute_view(const Shape& screen) const
{
    Mat4 view = Mat4::Identity();
    Point2 screenLL, screenUR;
    if(screen.ratio<float>() < bounds_.shape().ratio<float>()) {
        screenLL.x = bounds_.left;
        screenUR.x = bounds_.right;
        screenLL.y = 0.5f*(bounds_.bottom + bounds_.top) 
                   - 0.5f*bounds_.width() / screen.ratio<float>();
        screenUR.y = 0.5f*(bounds_.bottom + bounds_.top) 
                   + 0.5f*bounds_.width() / screen.ratio<float>();
    }
    else {
        screenLL.y = bounds_.bottom;
        screenUR.y = bounds_.top;
        screenLL.x = 0.5f*(bounds_.left + bounds_.right)
                   - 0.5f*bounds_.height() * screen.ratio<float>();
        screenUR.x = 0.5f*(bounds_.left + bounds_.right)
                   + 0.5f*bounds_.height() * screen.ratio<float>();
    }
    return View::from_corners(screenLL, screenUR);
}

void FanRenderer::draw(const View::ConstPtr& view) const
{
    Mat4 mat = this->compute_view(view->screen_size());

    glUseProgram(renderProgram_);

    corners_.bind(GL_ARRAY_BUFFER);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram_, "view"),
        1, GL_FALSE, mat.data());

    glUniform2f(glGetUniformLocation(renderProgram_, "angleBounds"),
                angle_.min, angle_.max);
    glUniform2f(glGetUniformLocation(renderProgram_, "rangeBounds"),
                range_.min, range_.max);

    glUniform1i(glGetUniformLocation(renderProgram_, "colormap"), 0);
    glActiveTexture(GL_TEXTURE0);
    colormap_->texture().bind(GL_TEXTURE_2D);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnableVertexAttribArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    GL_CHECK_LAST();
}


}; //namespace display
}; //namespace rtac
