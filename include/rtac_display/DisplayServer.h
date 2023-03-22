#ifndef _DEF_RTAC_DISPLAY_DISPLAY_SERVER_H_
#define _DEF_RTAC_DISPLAY_DISPLAY_SERVER_H_

#include <memory>
#include <deque>
#include <thread>
#include <unordered_map>
#include <functional>

#include <rtac_base/async/AsyncFunction.h>
#include <rtac_base/async/AsyncWorker.h>

#include <rtac_display/Display.h>

namespace rtac { namespace display {

/**
 * The main purpose of this type is to run OpenGL context in a thread separated
 * from the main thread, using a rtac::AsyncWorker for synchronous calls to
 * OpenGL API from another thread.
 *
 * It also allows to dissociate the working/user thread from the display
 * thread. This allows for higher frame rate on display side.
 */
class DisplayServer
{
    public:

    using Ptr      = std::unique_ptr<DisplayServer>;
    using ConstPtr = std::unique_ptr<const DisplayServer>;

    private:

    static std::unique_ptr<DisplayServer> instance_;
    static std::mutex creationMutex_;

    protected:

    AsyncWorker worker_;
    std::thread mainThread_;
    std::mutex  mainMutex_;
    bool        shouldStop_;
    bool        isRunning_;
    bool        blockAtEnd_;

    std::deque<std::unique_ptr<Display>> displays_;

    void run();
    void draw();

    DisplayServer(bool blockAtEnd = true);

    public:

    ~DisplayServer();
    static DisplayServer* Get();

    bool should_stop() const;

          Display& display(unsigned int idx)       { return *displays_[idx]; }
    const Display& display(unsigned int idx) const { return *displays_[idx]; }

    template <class F, class... Args>
    auto execute(F f, Args&&... args);

    template <class D, class... Args>
    D* create_display(Args&&... args);
};

template <class F, class... Args>
auto DisplayServer::execute(F f, Args&&... args)
{
    return worker_.execute(f, std::forward<Args>(args)...);
}

template <class D, class... Args>
D* DisplayServer::create_display(Args&&... args)
{
    return this->execute([&]() {
        auto display = std::make_unique<D>(std::forward<Args>(args)...);
        D* res = display.get();
        displays_.push_back(std::move(display));
        return res;
    });
}

} //namespace display
} //namespace rtac

#endif //_DEF_RTAC_DISPLAY_DISPLAY_SERVER_H_
