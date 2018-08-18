#define BOOST_TEST_MODULE "test_datasetManager"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <iostream>
#include "json.hpp"
#include "visUtil.hpp"

// the code to test:
#include "datasetManager.hpp"

using json = nlohmann::json;

using namespace std::string_literals;

struct TestContext {
    TestContext() { BOOST_TEST_MESSAGE( "setup fixture" ); }
    ~TestContext()               { BOOST_TEST_MESSAGE( "teardown fixture" ); }

    void check_equal(const vector<input_ctype>& a, const vector<input_ctype>& b) {
        auto ita = a.begin();
        auto itb = b.begin();

        while(ita != a.end() || itb != b.end())
        {
            BOOST_CHECK_EQUAL(ita->chan_id, itb->chan_id);
            BOOST_CHECK_EQUAL(ita->correlator_input, itb->correlator_input);
            if(ita != a.end())
            {
                ++ita;
            }
            if(itb != b.end())
            {
                ++itb;
            }
        }
    }
};

BOOST_FIXTURE_TEST_CASE( _general, TestContext ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    // generate datasets:
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs));
    dset_id init_ds_id = dm.add_dataset(input_state.first, -1);


    // transform that data:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"),
                                           input_ctype(2, "2"),
                                           input_ctype(3, "3"),
                                           input_ctype(4, "4")};
    pair<dset_id, const inputState*> old_state =
            dm.closest_ancestor_of_type<inputState>(init_ds_id);
    const std::vector<input_ctype>& old_inputs = old_state.second->get_inputs();
    check_equal(old_inputs, inputs);
    std::pair<state_id, const inputState*> transformed_input_state =
            dm.add_state(std::make_unique<inputState>(new_inputs));
    dset_id transformed_ds_id = dm.add_dataset(transformed_input_state.first,
                                               old_state.first);


    // get state
    pair<dset_id, const inputState*> final_state =
            dm.closest_ancestor_of_type<inputState>(transformed_ds_id);
    const std::vector<input_ctype>& final_inputs = final_state.second->get_inputs();
    check_equal(new_inputs, final_inputs);
}

BOOST_AUTO_TEST_CASE( _serialization_input ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    // serialize and deserialize
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs));
    json j = input_state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id, const inputState*> input_state3 =
            dm.add_state(std::make_unique<inputState>(inputs));
    json j3 = input_state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<input_ctype> diff_inputs = {input_ctype(1, "1"),
                                            input_ctype(7, "7")};
    std::pair<state_id, const inputState*> diff_input_state =
            dm.add_state(std::make_unique<inputState>(diff_inputs));
    json j_diff = diff_input_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}
