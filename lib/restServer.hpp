#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include "Config.hpp"
#include "mongoose.h"
#include "json.hpp"

#include <thread>

class restServer {

public:
    restServer();
    virtual ~restServer();

    void start();

    void mongoose_thread();

    static void handle_status(struct mg_connection *nc, int ev, void *ev_data);
    static void handle_notfound(struct mg_connection *nc, int ev, void *ev_data);

private:
    struct mg_mgr mgr;
    struct mg_connection *nc;
    const char *port = "12048";

    std::thread main_thread;
};

restServer * get_rest_server();

#endif /* CHIME_HPP */