#ifndef _DEF_RTAC_DISPLAY_GL_STATE_H_
#define _DEF_RTAC_DISPLAY_GL_STATE_H_

#include <cstring>
#include <unordered_map>

#include <rtac_base/types/Handle.h>

#include <rtac_display/utils.h>
#include <rtac_display/GLState.h>

namespace rtac { namespace display {

struct UnknownCapability : public std::exception
{
    std::string message_;

    UnknownCapability(GLenum cap) {
        std::ostringstream oss;
        oss << "Unknown OpenGL capability : " << cap;
        message_ = oss.str();
    }

    const char* what() const throw() {
        return message_.c_str();
    }
};

class GLState
{
    public:

    using Ptr      = rtac::Handle<GLState>;
    using ConstPtr = rtac::Handle<const GLState>;
    using StateMap = std::unordered_map<GLenum, bool>;

    protected:

    StateMap stateMap_;

    public:

    GLState();

    static Ptr New() { return Ptr(new GLState()); } // deprecated
    static Ptr Create() { return Ptr(new GLState()); }

    void enable(GLenum cap) {
        bool& state = this->capability(cap);
        if(!state) {
            glEnable(cap);
            state = true;
        }
    }

    void disable(GLenum cap) {
        bool& state = this->capability(cap);
        if(state) {
            glDisable(cap);
            state = false;
        }
    }

    bool is_enabled(GLenum cap) const
    {
        auto it = stateMap_.find(cap);
        if(it == stateMap_.end())
            throw UnknownCapability(cap);
        else
            return it->second;
    }

    protected:

    bool& capability(GLenum cap)
    {
        auto it = stateMap_.find(cap);
        if(it == stateMap_.end())
            throw UnknownCapability(cap);
        else
            return it->second;
    }

    public:

    static constexpr std::array<GLenum,2> DefaultTrueStates = {
        GL_DITHER,
        GL_MULTISAMPLE,
    };

    static constexpr std::array<GLenum,89> StateNames = {
        GL_ALPHA_TEST,
        GL_AUTO_NORMAL,
        GL_BLEND,
        GL_CLIP_PLANE0,
        GL_CLIP_PLANE0 + 1,
        GL_CLIP_PLANE0 + 2,
        GL_CLIP_PLANE0 + 3,
        GL_CLIP_PLANE0 + 4,
        GL_CLIP_PLANE0 + 5,
        GL_CLIP_PLANE0 + 6,
        GL_CLIP_PLANE0 + 7,
        GL_COLOR_LOGIC_OP,
        GL_COLOR_MATERIAL,
        GL_COLOR_SUM,
        GL_COLOR_TABLE,
        GL_CONVOLUTION_1D,
        GL_CONVOLUTION_2D,
        GL_CULL_FACE,
        GL_DEPTH_TEST,
        GL_DITHER,
        GL_FOG,
        GL_HISTOGRAM,
        GL_INDEX_LOGIC_OP,
        GL_LIGHT0,
        GL_LIGHT0 + 1,
        GL_LIGHT0 + 2,
        GL_LIGHT0 + 3,
        GL_LIGHT0 + 4,
        GL_LIGHT0 + 5,
        GL_LIGHT0 + 6,
        GL_LIGHT0 + 7,
        GL_LIGHT0 + 8,
        GL_LIGHT0 + 9,
        GL_LIGHT0 + 10,
        GL_LIGHT0 + 11,
        GL_LIGHT0 + 12,
        GL_LIGHT0 + 13,
        GL_LIGHT0 + 14,
        GL_LIGHT0 + 15,
        GL_LIGHTING,
        GL_LINE_SMOOTH,
        GL_LINE_STIPPLE,
        GL_MAP1_COLOR_4,
        GL_MAP1_INDEX,
        GL_MAP1_NORMAL,
        GL_MAP1_TEXTURE_COORD_1,
        GL_MAP1_TEXTURE_COORD_2,
        GL_MAP1_TEXTURE_COORD_3,
        GL_MAP1_TEXTURE_COORD_4,
        GL_MAP1_VERTEX_3,
        GL_MAP1_VERTEX_4,
        GL_MAP2_COLOR_4,
        GL_MAP2_INDEX,
        GL_MAP2_NORMAL,
        GL_MAP2_TEXTURE_COORD_1,
        GL_MAP2_TEXTURE_COORD_2,
        GL_MAP2_TEXTURE_COORD_3,
        GL_MAP2_TEXTURE_COORD_4,
        GL_MAP2_VERTEX_3,
        GL_MAP2_VERTEX_4,
        GL_MINMAX,
        GL_MULTISAMPLE,
        GL_NORMALIZE,
        GL_POINT_SMOOTH,
        GL_POINT_SPRITE,
        GL_POLYGON_OFFSET_FILL,
        GL_POLYGON_OFFSET_LINE,
        GL_POLYGON_OFFSET_POINT,
        GL_POLYGON_SMOOTH,
        GL_POLYGON_STIPPLE,
        GL_POST_COLOR_MATRIX_COLOR_TABLE,
        GL_POST_CONVOLUTION_COLOR_TABLE,
        GL_RESCALE_NORMAL,
        GL_SAMPLE_ALPHA_TO_COVERAGE,
        GL_SAMPLE_ALPHA_TO_ONE,
        GL_SAMPLE_COVERAGE,
        GL_SEPARABLE_2D,
        GL_SCISSOR_TEST,
        GL_STENCIL_TEST,
        GL_TEXTURE_1D,
        GL_TEXTURE_2D,
        GL_TEXTURE_3D,
        GL_TEXTURE_CUBE_MAP,
        GL_TEXTURE_GEN_Q,
        GL_TEXTURE_GEN_R,
        GL_TEXTURE_GEN_S,
        GL_TEXTURE_GEN_T,
        GL_VERTEX_PROGRAM_POINT_SIZE,
        GL_VERTEX_PROGRAM_TWO_SIDE,
    };
};



}; //namespace display
}; //namespace rtac


#endif //_DEF_RTAC_DISPLAY_GL_STATE_H_

