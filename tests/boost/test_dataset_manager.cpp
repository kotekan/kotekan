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
    void check_equal(const vector<input_ctype>& a, const vector<input_ctype>& b) {
        auto ita = a.begin();
        auto itb = b.begin();

        while(ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->chan_id, itb->chan_id);
            BOOST_CHECK_EQUAL(ita->correlator_input, itb->correlator_input);
            if(ita != a.end())
                ++ita;
            if(itb != b.end())
                ++itb;
        }
    }

    void check_equal(const vector<prod_ctype>& a, const vector<prod_ctype>& b) {
        auto ita = a.begin();
        auto itb = b.begin();

        while(ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->input_a, itb->input_a);
            BOOST_CHECK_EQUAL(ita->input_b, itb->input_b);
            if(ita != a.end())
                ++ita;
            if(itb != b.end())
                ++itb;
        }
    }

    void check_equal(const std::vector<std::pair<uint32_t, freq_ctype>>& a,
                     const std::vector<std::pair<uint32_t, freq_ctype>>& b) {
        auto ita = a.begin();
        auto itb = b.begin();

        while(ita != a.end() || itb != b.end()) {
            BOOST_CHECK_EQUAL(ita->first, itb->first);
            BOOST_CHECK_EQUAL(ita->second.centre, itb->second.centre);
            BOOST_CHECK_EQUAL(ita->second.width, itb->second.width);
            if(ita != a.end())
                ++ita;
            if(itb != b.end())
                ++itb;
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
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};
    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));
    dset_id init_ds_id = dm.add_dataset(input_state.first, -1);
    inputs = {input_ctype(1, "1"),
              input_ctype(2, "2")};
    prods = {{1, 1},
             {2, 2}};
    freqs = {{1, {1.1, 1}},
             {2, {2, 2.2}}};
    std::pair<state_id, const inputState*>input_state2 =
            dm.add_state(std::make_unique<inputState>(inputs,
                              make_unique<prodState>(prods,
                              make_unique<freqState>(freqs))));
    dset_id init_ds_id2 = dm.add_dataset(input_state2.first, init_ds_id);


    // transform that data:
    std::vector<input_ctype> new_inputs = {input_ctype(1, "1"),
                                           input_ctype(2, "2"),
                                           input_ctype(3, "3"),
                                           input_ctype(4, "4")};
    std::vector<prod_ctype> new_prods = {{1, 1},
                                         {2, 2},
                                         {3, 3},
                                         {4, 4}};
    std::vector<std::pair<uint32_t, freq_ctype>> new_freqs = {{1, {1.1, 1}},
                                                              {2, {2, 2.2}},
                                                              {3, {3, 3}},
                                                              {4, {4, 4}}};
    pair<dset_id, const inputState*> old_state =
            dm.closest_ancestor_of_type<inputState>(init_ds_id2);
    const std::vector<input_ctype>& old_inputs = old_state.second->get_inputs();
    check_equal(old_inputs, inputs);
    pair<dset_id, const prodState*> old_state2 =
         dm.closest_ancestor_of_type<prodState>(init_ds_id2);
    const std::vector<prod_ctype>& old_prods = old_state2.second->get_prods();
    check_equal(old_prods, prods);

    pair<dset_id, const freqState*> old_state3 =
         dm.closest_ancestor_of_type<freqState>(init_ds_id2);
    const std::vector<std::pair<uint32_t, freq_ctype>>& old_freqs =
            old_state3.second->get_freqs();
    check_equal(old_freqs, freqs);

    std::pair<state_id, const inputState*> transformed_input_state =
            dm.add_state(std::make_unique<inputState>(new_inputs,
                                           make_unique<prodState>(new_prods,
                                           make_unique<freqState>(new_freqs))));
    dset_id transformed_ds_id = dm.add_dataset(transformed_input_state.first,
                                               old_state.first);


    // get state
    pair<dset_id, const inputState*> final_state =
            dm.closest_ancestor_of_type<inputState>(transformed_ds_id);
    const std::vector<input_ctype>& final_inputs = final_state.second->get_inputs();
    check_equal(new_inputs, final_inputs);

    std::cout << dm.summary() << std::endl;

    for (auto s : dm.states())
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;

    for (auto s : dm.datasets())
        std::cout << s.second.first << " - " << s.second.second << std::endl;

    for (auto s : dm.ancestors(transformed_ds_id))
        std::cout << s.first << " - " << s.second->data_to_json().dump()
                  << std::endl;
}

BOOST_AUTO_TEST_CASE( _serialization_input ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    // serialize and deserialize
    std::vector<input_ctype> inputs = {input_ctype(1, "1"),
                                       input_ctype(2, "2"),
                                       input_ctype(3, "3")};
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};
    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));
    json j = input_state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id, const inputState*> input_state3 =
            dm.add_state(std::make_unique<inputState>(inputs,
                                               make_unique<prodState>(prods,
                                               make_unique<freqState>(freqs))));
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

BOOST_AUTO_TEST_CASE( _serialization_prod ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    // serialize and deserialize
    std::vector<prod_ctype> prods = {{1, 1},
                                     {2, 2},
                                     {3, 3}};
    std::pair<state_id, const prodState*> state =
            dm.add_state(std::make_unique<prodState>(prods));
    json j = state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id, const prodState*> state3 =
            dm.add_state(std::make_unique<prodState>(prods));
    json j3 = state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<prod_ctype> diff_prods = {{1, 2},
                                          {7, 3}};
    std::pair<state_id, const prodState*> diff_state =
            dm.add_state(std::make_unique<prodState>(diff_prods));
    json j_diff = diff_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}

BOOST_AUTO_TEST_CASE( _serialization_freq ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    // serialize and deserialize
    std::vector<std::pair<uint32_t, freq_ctype>> freqs = {{1, {1.1, 1}},
                                                          {2, {2, 2.2}},
                                                          {3, {3, 3}}};
    std::pair<state_id, const freqState*> state =
            dm.add_state(std::make_unique<freqState>(freqs));
    json j = state.second->to_json();
    state_uptr s = datasetState::from_json(j);
    json j2 = s->to_json();
    BOOST_CHECK_EQUAL(j, j2);

    // serialize 2 states with the same data
    std::pair<state_id, const freqState*> state3 =
            dm.add_state(std::make_unique<freqState>(freqs));
    json j3 = state3.second->to_json();
    BOOST_CHECK_EQUAL(j, j3);

    // check that different data leads to different json
    std::vector<std::pair<uint32_t, freq_ctype>> diff_freqs = {{1, {1.2, 2}},
                                                               {3, {7, 3}}};
    std::pair<state_id, const freqState*> diff_state =
            dm.add_state(std::make_unique<freqState>(diff_freqs));
    json j_diff = diff_state.second->to_json();
    BOOST_CHECK(j != j_diff);
}

BOOST_AUTO_TEST_CASE( _no_state_of_type_found ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    std::vector<input_ctype> inputs = {input_ctype(1, "1")};
    std::pair<state_id, const inputState*> input_state =
            dm.add_state(std::make_unique<inputState>(inputs));
    dset_id init_ds_id = dm.add_dataset(input_state.first, -1);

    std::pair<dset_id, const prodState*> not_found =
            dm.closest_ancestor_of_type<prodState>(init_ds_id);

    BOOST_CHECK_EQUAL(not_found.first, -1);
    BOOST_CHECK_EQUAL(not_found.second, // == nullptr
                      static_cast<decltype(not_found.second)>(nullptr));
}
