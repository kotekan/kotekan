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
 * have a timestamp later than the one associated to this update". The queue is
 * ordered by the updates timestamps.
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
        _len = len;

        while (values.size() > _len)
            values.pop_front();
    }

    /**
     * @brief Get the current size of the queue.
     *
     * This may be less than the length if not enough updates have been posted.
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
        // usually just push update to the back of the queue
        if (!values.size() || timestamp > values.crbegin()->first)
            values.push_back(std::pair<timespec, T>(timestamp, std::move(update)));
        else { // this is more complicated...
            auto u = values.rbegin();
            while (u->first > timestamp) {
                u++;
                if (u == values.crend())
                    break;
            }
            // check if timestamp is identical -> replace update

            if (u != values.crend() && u->first == timestamp)
                u->second = std::move(update);
            else
                // insert the new update where it belongs in the queue
                values.insert(u.base(), std::pair<timespec, T>(timestamp, std::move(update)));
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
    std::pair<timespec, const T*> get_update(timespec timestamp) {
        auto u = values.crbegin();

        while (u != values.crend() && u->first > timestamp) {
            u++;
        }

        if (u == values.crend()) {
            return std::pair<timespec, const T*>({0, 0}, nullptr);
        }

        return std::pair<timespec, const T*>(u->first, &(u->second));
    };

private:
    // The updates with their timestamps ("use this value for frames with
    // timestamps later than this").
    std::deque<std::pair<timespec, T>> values;

    // Length of the queue.
    size_t _len;
};

#endif // UPDATEQUEUE_HPP
