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
uniform sampler2D fanData;
uniform sampler2D colormap;

uniform vec2 valueScaling;
uniform vec2 angleScaling;
uniform vec2 rangeScaling;
uniform vec2 angleBounds;
uniform vec2 rangeBounds;

out vec4 outColor;

#define M_2PI 6.283185307179586

void main()
{
    float r     = length(xyPos);
    float theta = atan(xyPos.y, xyPos.x);
    if(theta < angleBounds.x)
        theta += M_2PI;

    if(theta < angleBounds.x || theta > angleBounds.y ||
       r     < rangeBounds.x || r     > rangeBounds.y)
    {
        outColor = texture(colormap, vec2(0,0));
    }
    else {
        vec2 p;
        p.x = 1.0 - fma(angleScaling.x, theta,  angleScaling.y);
        p.y = fma(rangeScaling.x, r,      rangeScaling.y);
        float value = fma(valueScaling.x, texture(fanData,p).x, valueScaling.y);
        outColor = texture(colormap, vec2(value, 0.0f));
    }
}
)");

const std::string& FanRenderer::fragmentShaderNonLinear = std::string(R"(
#version 430 core

in      vec2 xyPos;
uniform sampler2D fanData;
uniform sampler2D colormap;
uniform sampler2D bearingMap;

uniform vec2 valueScaling;
uniform vec2 angleScaling;
uniform vec2 rangeScaling;
uniform vec2 angleBounds;
uniform vec2 rangeBounds;

out vec4 outColor;

#define M_2PI 6.283185307179586

void main()
{
    float r     = length(xyPos);
    float theta = atan(xyPos.y, xyPos.x);
    if(theta < angleBounds.x)
        theta += M_2PI;

    if(theta < angleBounds.x || theta > angleBounds.y ||
       r     < rangeBounds.x || r     > rangeBounds.y)
    {
        outColor = texture(colormap, vec2(0,0));
    }
    else {
        vec2 p;
        p.x = fma(angleScaling.x, theta,  angleScaling.y);
        p.y = fma(rangeScaling.x, r,      rangeScaling.y);
        p.x = 1.0f - texture(bearingMap, vec2(p.x,0.0)).x;
        float value = fma(valueScaling.x, texture(fanData,p).x, valueScaling.y);
        outColor = texture(colormap, vec2(value, 0.0f));
    }
}
)");

FanRenderer::FanRenderer(const GLContext::Ptr& context) :
    Renderer(context, vertexShader, fragmentShader),
    data_(GLTexture::Create()),
    colormap_(Colormap::Create(colormap::Viridis())),
    valueRange_({0.0f,1.0f}),
    angle_({-M_PI, M_PI}),
    range_({0.0f,1.0f}),
    corners_(6),
    direction_(Direction::Down),
    linearBearingsProgram_(renderProgram_),
    nonlinearBearingsProgram_(create_render_program(vertexShader, fragmentShaderNonLinear))
{
    this->set_geometry(angle_, range_);
    //data_->set_filter_mode(GLTexture::FilterMode::Nearest);
    data_->set_filter_mode(GLTexture::FilterMode::Linear);

    //data_->set_wrap_mode(GLTexture::WrapMode::Clamp);
    data_->set_wrap_mode(GLTexture::WrapMode::Mirror);
}

FanRenderer::Ptr FanRenderer::Create(const GLContext::Ptr& context)
{
    return Ptr(new FanRenderer(context));
}

void FanRenderer::set_value_range(Interval valueRange)
{
    if(fabs(valueRange.length()) < 1.0e-6)
        return;
    valueRange_ = valueRange;
}

void FanRenderer::set_geometry_degrees(const Interval& angle, const Interval& range)
{
    this->set_geometry({(float)(angle.lower * M_PI / 180.0f),
                        (float)(angle.upper * M_PI / 180.0f)},
                       range);
}

void FanRenderer::set_geometry(Interval angle, const Interval& range)
{
    using Point2 = rtac::Point2<float>;

    // Normalizing angle bounds
    // (angle.lower in [-pi,pi] and angle.upper in ]angle.lower, angle.lower + 2pi])
    while(angle.upper > angle.lower + 2*M_PI) angle.upper -= 2*M_PI;
    while(angle.upper <= angle.lower + 0.01f) angle.upper += 2*M_PI;
    while(angle.lower >  M_PI) {
        angle.lower -= 2*M_PI;
        angle.upper -= 2*M_PI;
    }
    while(angle.lower < -M_PI) { 
        angle.lower += 2*M_PI;
        angle.upper += 2*M_PI;
    }

    // finding extermas of the fan display area.
    std::vector<Point2> poi;
    poi.push_back({range.lower*std::cos(angle.lower), range.lower*std::sin(angle.lower)});
    poi.push_back({range.lower*std::cos(angle.upper), range.lower*std::sin(angle.upper)});
    poi.push_back({range.upper*std::cos(angle.lower), range.upper*std::sin(angle.lower)});
    poi.push_back({range.upper*std::cos(angle.upper), range.upper*std::sin(angle.upper)});

    auto is_inside = [](float angle, const Interval& bounds) { 
        if(angle < bounds.lower) angle += 2*M_PI;
        return angle <= bounds.upper; 
    };
    if(is_inside(0.0f, angle)) {
        poi.push_back({range.lower, 0.0});
        poi.push_back({range.upper, 0.0});
    }
    if(is_inside(M_PI, angle)) {
        poi.push_back({-range.lower, 0.0});
        poi.push_back({-range.upper, 0.0});
    }
    if(is_inside(0.5f*M_PI, angle)) {
        poi.push_back({0.0, range.lower});
        poi.push_back({0.0, range.upper});
    }
    if(is_inside(-0.5f*M_PI, angle)) {
        poi.push_back({0.0, -range.lower});
        poi.push_back({0.0, -range.upper});
    }

    bounds_.left   = poi[0].x;
    bounds_.right  = poi[0].x;
    bounds_.top    = poi[0].y;
    bounds_.bottom = poi[0].y;
    for(auto p : poi) {
        bounds_.left   = std::min(bounds_.left,   p.x);
        bounds_.right  = std::max(bounds_.right,  p.x);
        bounds_.top    = std::max(bounds_.top,    p.y);
        bounds_.bottom = std::min(bounds_.bottom, p.y);
    }

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

void FanRenderer::set_aperture(Interval angle)
{
    this->set_geometry(angle, range_);
}

void FanRenderer::set_range(Interval range)
{
    this->set_geometry(angle_, range);
}

void FanRenderer::set_data(const GLTexture::Ptr& tex)
{
    data_ = tex;
}

void FanRenderer::set_data(const Shape& shape, const float* data)
{
    data_->set_image(shape, data);
}

void FanRenderer::set_data(const Shape& shape, const GLVector<float>& data,
                           bool computeScale)
{
    data_->set_image(shape, data);
    if(computeScale)
        this->compute_scale(data);
}

void FanRenderer::set_bearings(unsigned int nBeams, const float* bearings,
                               unsigned int mapSize)
{
    Interpolator::Vector x0(nBeams);
    Interpolator::Vector y0(nBeams);
    
    for(int i = 0; i < nBeams; i++) {
        //x0[i] = ((float)i) / (nBeams - 1);
        // opengl textures coordinates and defined on [1/2N, 1 - 1/2N]
        x0[i] = (i + 0.5f) / nBeams;
        y0[i] = bearings[i];
    }

    if(!mapSize)
        mapSize = nBeams;
    Interpolator::Vector b(mapSize);
    for(int i = 0; i < mapSize; i++) {
        b[i]  = ((bearings[nBeams-1] - bearings[0])*i) / (mapSize - 1) + bearings[0];
    }

    auto ib = Interpolator::CreateCubicSpline(y0,x0)(b);

    if(!bearingMap_) {
        bearingMap_ = GLTexture::Create();
        bearingMap_->set_filter_mode(GLTexture::FilterMode::Linear);
        bearingMap_->set_wrap_mode(GLTexture::WrapMode::Clamp);
        GL_CHECK_LAST();
    }
    bearingMap_->set_image({(unsigned int)mapSize, 1}, ib.data());
    this->enable_bearing_map();
    this->set_aperture({bearings[0], bearings[nBeams-1]});
}

void FanRenderer::enable_bearing_map()
{
    if(bearingMap_)
        renderProgram_ = nonlinearBearingsProgram_;
}

void FanRenderer::disable_bearing_map()
{
    renderProgram_ = linearBearingsProgram_;
}

FanRenderer::Mat4 FanRenderer::compute_view(const Shape& screen) const
{
    Mat4 rotation = Mat4::Identity();
    auto bounds = bounds_;
    switch(direction_) {
        case Direction::Left:
            rotation(0,0) = -1;
            rotation(1,1) = -1;
            bounds.left   = -bounds_.right;
            bounds.bottom = -bounds_.top;
            bounds.right  = -bounds_.left;
            bounds.top    = -bounds_.bottom;
            break;
        case Direction::Up:
            rotation(0,0) = 0;
            rotation(1,1) = 0;
            rotation(1,0) = 1;
            rotation(0,1) = -1;
            bounds.left   = -bounds_.top;
            bounds.bottom =  bounds_.left;
            bounds.right  = -bounds_.bottom;
            bounds.top    =  bounds_.right;
            break;
        case Direction::Down:
            rotation(0,0) = 0;
            rotation(1,1) = 0;
            rotation(1,0) = -1;
            rotation(0,1) = 1;
            bounds.left   =  bounds_.bottom;
            bounds.bottom = -bounds_.right;
            bounds.right  =  bounds_.top;
            bounds.top    = -bounds_.left;
            break;
    }
    Point2 screenLL, screenUR;
    if(screen.ratio<float>() < bounds.shape().ratio<float>()) {
        screenLL.x = bounds.left;
        screenUR.x = bounds.right;
        screenLL.y = 0.5f*(bounds.bottom + bounds.top) 
                   - 0.5f*bounds.width() / screen.ratio<float>();
        screenUR.y = 0.5f*(bounds.bottom + bounds.top) 
                   + 0.5f*bounds.width() / screen.ratio<float>();
    }
    else {
        screenLL.y = bounds.bottom;
        screenUR.y = bounds.top;
        screenLL.x = 0.5f*(bounds.left + bounds.right)
                   - 0.5f*bounds.height() * screen.ratio<float>();
        screenUR.x = 0.5f*(bounds.left + bounds.right)
                   + 0.5f*bounds.height() * screen.ratio<float>();
    }
    return View::from_corners(screenLL,screenUR)*rotation;
}

void FanRenderer::compute_scale(const GLVector<float>& data)
{
    this->set_value_range({reductor_.min(data),
                           reductor_.max(data)});
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

    glUniform2f(glGetUniformLocation(renderProgram_, "valueScaling"),
                1.0f / valueRange_.length(), -valueRange_.lower / valueRange_.length());
    glUniform2f(glGetUniformLocation(renderProgram_, "angleScaling"),
                1.0f / angle_.length(), -angle_.lower / angle_.length());
    glUniform2f(glGetUniformLocation(renderProgram_, "rangeScaling"),
                1.0f / range_.length(), -range_.lower / range_.length());
    glUniform2f(glGetUniformLocation(renderProgram_, "angleBounds"),
                angle_.lower, angle_.upper);
    glUniform2f(glGetUniformLocation(renderProgram_, "rangeBounds"),
                range_.lower, range_.upper);

    glUniform1i(glGetUniformLocation(renderProgram_, "fanData"), 0);
    glActiveTexture(GL_TEXTURE0);
    data_->bind(GL_TEXTURE_2D);

    glUniform1i(glGetUniformLocation(renderProgram_, "colormap"), 1);
    glActiveTexture(GL_TEXTURE1);
    colormap_->texture().bind(GL_TEXTURE_2D);

    if(renderProgram_ == nonlinearBearingsProgram_ && bearingMap_) {
        glUniform1i(glGetUniformLocation(renderProgram_, "bearingMap"), 2);
        glActiveTexture(GL_TEXTURE2);
        bearingMap_->bind(GL_TEXTURE_2D);
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glEnableVertexAttribArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    GL_CHECK_LAST();
}


}; //namespace display
}; //namespace rtac
