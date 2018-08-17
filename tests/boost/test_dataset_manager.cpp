#define BOOST_TEST_MODULE "test_datasetManager"

#include <boost/test/included/unit_test.hpp>
#include <string>
#include <iostream>

#include "json.hpp"

// the code to test:
#include "datasetManager.hpp"

using json = nlohmann::json;

using namespace std::string_literals;

BOOST_AUTO_TEST_CASE( _general ) {
    __log_level = 4;
    datasetManager& dm = datasetManager::instance();

    json j;

    std::vector<uint32_t> ids(3,0);

    std::cout << "Test calling freqState constructor" << endl;
    state_uptr dt1 = std::make_unique<freqState>();
    std::cout << "Test calling freqState constructor" << endl;
    state_uptr dt2 = std::make_unique<freqState>(j, std::move(dt1));
    std::cout << dt2->to_json().dump() << std::endl;

    json j2 = dt2->to_json();
    state_uptr dt3 = datasetState::from_json(j2);
    std::cout << dt3->to_json().dump() << std::endl;

    std::pair<state_id, const freqState*> pair1 = dm.add_state(std::make_unique<freqState>(ids));
    state_id t1 = pair1.first;
    std::pair<state_id, const inputState*> pair2 = dm.add_state(std::make_unique<inputState>());
    state_id t2 = pair2.first;
    std::pair<state_id, const freqState*> pair3 = dm.add_state(std::make_unique<freqState>(j, std::move(dt2)));
    state_id t3 = pair3.first;

    dset_id d1 = dm.add_dataset(t1, -1);
    dset_id d2 = dm.add_dataset(t2, d1);
    dset_id d4 = dm.add_dataset(t3, d2);

    for (auto t : dm.ancestors(d4)) {
        std::cout << t.first << " " << *(t.second) << std::endl;\

    }

    auto ancestor = dm.ancestors(d4);
    BOOST_CHECK_EQUAL(ancestor[0].first, d4);
    BOOST_CHECK_EQUAL(ancestor[0].second->to_json(), j2);
    BOOST_CHECK_EQUAL(ancestor[1].first, d4);
    BOOST_CHECK_EQUAL(ancestor[2].first, d2);
    BOOST_CHECK_EQUAL(ancestor[3].first, d1);



    auto t = dm.closest_ancestor_of_type<inputState>(d4);
    std::cout << "Ancestor(input) " << t.first << " " << *(t.second) << std::endl;

    std::cout << dm.summary();
}
