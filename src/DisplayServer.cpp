#include <rtac_display/DisplayServer.h>

namespace rtac { namespace display {

DisplayServer::DisplayServer(bool blockAtEnd) :
    shouldStop_(false),
    isRunning_(false),
    blockAtEnd_(blockAtEnd)
{
    auto waiter = worker_.push_front(empty_async());
    mainThread_ = std::move(std::thread(std::bind(&DisplayServer::run, this)));
    if(!mainThread_.joinable()) {
        throw std::runtime_error("Fatal : could not start DisplayServer working thread.");
    }
    waiter.wait();
}

DisplayServer::~DisplayServer()
{
    if(!blockAtEnd_) {
        worker_.execute([&]() { displays_.clear(); });
    }
    if(isRunning_) {
        shouldStop_ = true;
        mainThread_.join();
    }
}

DisplayServer::Ptr DisplayServer::Create()
{
    return Ptr(new DisplayServer());
}

void DisplayServer::run()
{
    shouldStop_ = false;
    isRunning_  = true;
    while(!this->should_stop()) {
        worker_.run(); // this will empty callbacks in the worker
        this->draw();
    }
    isRunning_ = false;
}

void DisplayServer::draw()
{
    std::lock_guard<std::mutex> lock(mainMutex_);
    for(auto& disp : displays_) {
        disp->draw();
    }
}

bool DisplayServer::should_stop() const
{
    if(displays_.size() > 0) {
        for(const auto& d : displays_) {
            if(d->should_close()) {
                return true;
            }
        }
        return false;
    }
    return shouldStop_;
}

} //namespace display
} //namespace rtac
