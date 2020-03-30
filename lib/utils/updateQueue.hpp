#ifndef UPDATEQUEUE_HPP
#define UPDATEQUEUE_HPP

#include "visUtil.hpp"

#include <deque>

/**
 * @class updateQueue
 * @brief Class that keeps track of updates with timestamps in a FIFO
 *
 * This class wraps std::deque to keep updates and their timestamps in a FIFO
 * queue. A timestamp is understood as "apply this update to all frames that
 * have a timestamp later than (or equal to) the one associated to this
 * update". The queue is ordered by the updates timestamps. No future update
 * will ever be returned, that is if the only available updates have future
 * timestamps, nothing will be returned.
 *
 * @author Rick Nitsche
 */
template<class T>
class updateQueue {
public:
    /**
     * @brief Create an updateQueue of length len.
     *
     * @param len      The length of the queue.
     */
    updateQueue(const size_t len) : _len(len){};

    /**
     * @brief Create an updateQueue of length 0.
     */
    updateQueue() : _len(0){};

    /**
     * @brief Resize the queue.
     * @param len   The new size.
     */
    void resize(const size_t len) {
        // Lock the queue
        std::scoped_lock _lock(update_lock);

        _len = len;

        while (values.size() > _len)
            values.pop_front();
    }

    /**
     * @brief Get the current size of the queue.
     *
     * This may be less than the maximum size if not enough updates have been posted.
     *
     * @returns  size  The current size.
     **/
    size_t size() const {
        return values.size();
    }


    /**
     * @brief Insert an update.
     *
     * Inserts an update into the queue, according to its timestamp. Removes
     * The oldest update if otherwise the queue would exceed its length.
     *
     * @param timestamp      The timestamp of the inserted update.
     * @param update         The value of the inserted update.
     */
    void insert(timespec timestamp, T&& update) {

        // Lock the queue
        std::scoped_lock _lock(update_lock);

        // usually just push update to the back of the queue
        if (!values.size() || timestamp > values.crbegin()->first)
            values.push_back({timestamp, std::make_shared<const T>(std::move(update))});
        else { // this is more complicated...
            auto u = values.rbegin();
            while (u->first > timestamp) {
                u++;
                if (u == values.crend())
                    break;
            }
            // check if timestamp is identical -> replace update

            if (u != values.crend() && u->first == timestamp)
                u->second = std::make_shared<const T>(std::move(update));
            else
                // insert the new update where it belongs in the queue
                values.insert(u.base(), {timestamp, std::make_shared<const T>(std::move(update))});
        }

        if (values.size() > _len)
            values.pop_front();
    };

    /**
     * @brief Fetch the latest update for a frame with the given timestamp.
     *
     * Finds the update from the queue that should be applied to a frame
     * with the given timestamp.
     *
     * @param  timestamp  The timestamp of a frame.
     *
     * @returns  The value of the most recent update from before the given timestamp
     *           and the timestamp associated to the update. If the queue is empty, or
     *           all updates are in the future, a nullptr is returned as update.
     */
    std::pair<timespec, std::shared_ptr<const T>> get_update(timespec timestamp) const {

        // Lock the queue
        std::scoped_lock _lock(update_lock);

        auto u = values.crbegin();

        while (u != values.crend() && u->first > timestamp) {
            u++;
        }

        if (u == values.crend()) {
            return {{0, 0}, nullptr};
        }

        return *u;
    };

    /**
     * @brief Get all updates stored by the queue and their timestamps.
     *
     * @return A const reference to an std::deque holding all updates and their timestamps.
     */
    const std::deque<std::pair<timespec, std::shared_ptr<const T>>>& get_all_updates() const {
        return values;
    }

private:
    // The updates with their timestamps ("use this value for frames with
    // timestamps later than this").
    std::deque<std::pair<timespec, std::shared_ptr<const T>>> values;

    // Length of the queue.
    size_t _len;

    // A mutex to ensure updates and fetches from this FIFO are thread safe
    mutable std::mutex update_lock;
};

// Define a custom fmt formatter that prints the timestamps
template<typename T>
struct fmt::formatter<updateQueue<T>> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const updateQueue<T>& q, FormatContext& ctx) {
        auto it = q.get_all_updates().begin();
        auto pos = ctx.out();
        while (it != q.get_all_updates().end()) {
            pos = format_to(pos, "{:f} ", ts_to_double(it->first));
            *it++;
        }
        return pos;
    }
};

#endif // UPDATEQUEUE_HPP
