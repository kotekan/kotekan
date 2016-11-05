#ifndef REST_SERVER_HPP
#define REST_SERVER_HPP

#include "Config.hpp"
#include "mongoose.h"
#include "json.hpp"

#include <thread>
#include <functional>
#include <map>

using json = nlohmann::json;
using std::map;

class restServer {

public:
    restServer();
    virtual ~restServer();

    void start();

    void mongoose_thread();

    static void handle_status(struct mg_connection *nc, int ev, void *ev_data);
    static void handle_packet_grab(struct mg_connection *nc, int ev, void *ev_data);
    static void handle_notfound(struct mg_connection *nc, int ev, void *ev_data);
    static void handle_start(struct mg_connection *nc, int ev, void *ev_data);

    void register_packet_callback(std::function<uint8_t*(int, int&)> callback, int port);

    // Provides json object and returns 0 if loading kotekan is successful.
    // Returns -1 on failure and sets string to error message.
    void register_start_callback(std::function<int(json &, string&)> callback);

    std::map<int, std::function<uint8_t*(int, int&)> > packet_callbacks;
    std::function<int(json&, string&)> start_callback;

private:

    // Returns the json parsed object from med.body, in json_parse
    // Returns 0 if the body is valid json.
    // Else returns -1 if the body is invalid json, and sends error message to client in header.
    // Do not call any ms_send calls if this function fails.
    int handle_json(struct mg_connection *nc, int ev, void *ev_data, json &json_parse);

    struct mg_mgr mgr;
    struct mg_connection *nc;
    const char *port = "12048";

    std::thread main_thread;
};

restServer * get_rest_server();

#endif /* CHIME_HPP */