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
    datasetManager& dm = datasetManager::get();
    // state_id t1 = dm.add_transform(std::make_unique<datasetTransform>("trans1"));
    // state_id t2 = dm.add_transform(std::make_unique<inputTransform>("trans2"));
    // state_id t3 = dm.add_transform(std::make_unique<freqTransform>("trans3", "test"));

    json j;

    // state_uptr dt1 = datasetTransform::create("freqTransform", "trans3", j);
    // state_uptr dt2 = datasetTransform::create("freqTransform", "trans3", j, std::move(dt1));


    state_uptr dt1 = std::make_unique<freqTransform>("trans3", "hello2");
    state_uptr dt2 = std::make_unique<freqTransform>("trans3", "hello", std::move(dt1));
    std::cout << dt2->to_json().dump() << std::endl;

    json j2 = dt2->to_json();
    state_uptr dt3 = datasetTransform::from_json(j2);
    std::cout << dt3->to_json().dump() << std::endl;

    state_id t1 = dm.add_transform(std::make_unique<inputTransform>("trans1"));
    state_id t2 = dm.add_transform(std::make_unique<inputTransform>("trans2"));
    // state_id t1 = dm.add_transform(datasetTransform::create("inputTransform", "trans1", j));
    // state_id t2 = dm.add_transform(datasetTransform::create("inputTransform", "trans2", j));
    state_id t3 = dm.add_transform(std::move(dt2));

    dset_id d1 = dm.add_dataset(t1, -1);
    dset_id d2 = dm.add_dataset(t2, d1);
    //dset_id d3 = dm.add_dataset(t3, d1);
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

    auto t = dm.closest_ancestor_of_type<inputTransform>(d4);
    std::cout << "Ancestor(input) " << t.first << " " << *(t.second) << std::endl;

    std::cout << dm.summary();
}
