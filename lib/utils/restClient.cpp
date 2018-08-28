#include "restClient.hpp"
#include "errors.h"
#include "json.hpp"
#include "mongoose.h"
#include "restServer.hpp"
#include <iostream>

int restClient::_s_exit_flag = 0;
const char * restClient::_s_url = nullptr;

restReply restClient::send_json(const char *s_url,
                   const nlohmann::json *request) {
    struct mg_mgr mgr;
    struct mg_connection *nc;
    std::string json_string = request->dump();
    struct restReply reply;

    _s_url = s_url;
    _s_exit_flag = 0;

    mg_mgr_init(&mgr, NULL);
    nc = mg_connect_http(&mgr, ev_handler, s_url, NULL, json_string.c_str());
    if (nc == NULL) {
        ERROR("restClient: Failed connecting to %s.", s_url);
        return reply;
    }
    mg_set_protocol_http_websocket(nc);

    DEBUG("restClient: Sent json_request to %s:\n%s", s_url,
          request->dump().c_str());
    while (_s_exit_flag == 0)
        mg_mgr_poll(&mgr, 1000);

    mg_mgr_free(&mgr);

    if (_s_exit_flag == -1)
        return reply;

    reply.success = true;
    return reply;
}

void restClient::ev_handler(struct mg_connection *nc, int ev, void *ev_data) {
  struct http_message *hm = (struct http_message *) ev_data;
  int connect_status;

  switch (ev) {
    case MG_EV_CONNECT:
      connect_status = *(int *) ev_data;
      if (connect_status != 0) {
        ERROR("restClient: Error connecting to %s: %s", _s_url, strerror(connect_status));
        _s_exit_flag = -1;
      }
      break;
    case MG_EV_HTTP_REPLY:
      if ((int) hm->resp_code != (int) HTTP_RESPONSE::OK) {
        ERROR("restClient: Got response:\n%s", hm->resp_status_msg);
        _s_exit_flag = -1;
        break;
      }
      nc->flags |= MG_F_SEND_AND_CLOSE;
      _s_exit_flag = 1;
      break;
    case MG_EV_CLOSE:
      if (_s_exit_flag == 0) {
        ERROR("restClient: Server closed connection");
        _s_exit_flag = -1;
      };
      break;
    default:
      break;
  }
}
