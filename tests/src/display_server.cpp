#include <iostream>
using namespace std;

#include <rtac_display/DisplayServer.h>
#include <rtac_display/samples/Display3D.h>
#include <rtac_display/renderers/Frame.h>
using namespace rtac::display;

int main()
{
    auto server = DisplayServer::Get();
    auto display = server->create_display<samples::Display3D>();
    auto renderer = server->execute([&]() { 
        return display->create_renderer<Frame>(display->view());
    });

    return 0;
}


