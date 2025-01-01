#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pcl/point_types.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <memory>
#include <thread>
#include <unistd.h>

namespace KdTree {

#define EPSS                           1e-6
#define Minimal_Unbalanced_Tree_Size   10
#define Multi_Thread_Rebuild_Point_Num 1500
#define DOWNSAMPLE_SWITCH              true
#define ForceRebuildPercentage         0.2
#define Q_LEN                          1000000

enum operationSet { ADD_POINT, DELETE_POINT, DELETE_BOX, ADD_BOX, DOWNSAMPLE_DELETE, PUSH_DOWN };
enum deletePointStorageSet { NOT_RECORD, DELETE_POINTS_REC, MULTI_THREAD_REC };

template <typename PointType>
class KdTree {
public:
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using Ptr         = std::shared_ptr<KdTree<PointType>>;

    explicit KdTree(float delete_param, float balance_param, float box_length) {
        delete_criterion_param  = delete_param;
        balance_criterion_param = balance_param;
        downsample_size         = box_length;
        termination_flag        = false;
        auto rebuildThread      = std::thread([this]() { rebuildCallBack(); });
        rebuildThread.detach();
    }

    struct BoxPointType {
        float vertex_min[3];
        float vertex_max[3];
    };

    struct KdNode {
        PointType point;
        int division_axis;
        int TreeSize                  = 1;
        int invalid_point_num         = 0;
        int down_del_num              = 0;
        bool point_deleted            = false;
        bool tree_deleted             = false;
        bool point_downsample_deleted = false;
        bool tree_downsample_deleted  = false;
        bool need_push_down_to_left   = false;
        bool need_push_down_to_right  = false;
        bool workingFlag              = false;
        std::mutex push_down_mutex_lock;
        // pthread_mutex_t push_down_mutex_lock;
        float node_range_x[2], node_range_y[2], node_range_z[2];
        float radius_sq;
        KdNode* leftSonPtr  = nullptr;
        KdNode* rightSonPtr = nullptr;
        KdNode* fatherPtr   = nullptr;
        // For paper data record
        float alpha_del;
        float alpha_bal;
    };

    struct Operation_Logger_Type {
        PointType point;
        BoxPointType boxpoint;
        bool tree_deleted, tree_downsample_deleted;
        operationSet op;
    };
    // static const PointType zeroP;

    struct PointType_CMP {
        PointType point;
        float dist = 0.0;
        explicit PointType_CMP(PointType p = PointType(), float d = INFINITY) {
            this->point = p;
            this->dist  = d;
        };
        bool operator<(const PointType_CMP& a) const {
            if (std::fabs(dist - a.dist) < 1e-10)
                return point.x < a.point.x;
            else
                return dist < a.dist;
        }
    };

    class MANUAL_HEAP {

    public:
        explicit MANUAL_HEAP(int max_capacity = 100) {
            cap       = max_capacity;
            heap      = new PointType_CMP[max_capacity];
            heap_size = 0;
        }

        ~MANUAL_HEAP() { delete[] heap; }
        void pop() {
            if (heap_size == 0)
                return;
            heap[0] = heap[heap_size - 1];
            heap_size--;
            MoveDown(0);
        }
        PointType_CMP top() { return heap[0]; }
        void push(PointType_CMP point) {
            if (heap_size >= cap)
                return;
            heap[heap_size] = point;
            FloatUp(heap_size);
            heap_size++;
        }
        int size() { return heap_size; }
        void clear() { heap_size = 0; }

    private:
        PointType_CMP* heap;
        void MoveDown(int heap_index) {
            int l             = heap_index * 2 + 1;
            PointType_CMP tmp = heap[heap_index];
            while (l < heap_size) {
                if (l + 1 < heap_size && heap[l] < heap[l + 1])
                    l++;
                if (tmp < heap[l]) {
                    heap[heap_index] = heap[l];
                    heap_index       = l;
                    l                = heap_index * 2 + 1;
                } else
                    break;
            }
            heap[heap_index] = tmp;
        }
        void FloatUp(int heap_index) {
            int ancestor      = (heap_index - 1) / 2;
            PointType_CMP tmp = heap[heap_index];
            while (heap_index > 0) {
                if (heap[ancestor] < tmp) {
                    heap[heap_index] = heap[ancestor];
                    heap_index       = ancestor;
                    ancestor         = (heap_index - 1) / 2;
                } else
                    break;
            }
            heap[heap_index] = tmp;
        }
        int heap_size = 0;
        int cap       = 0;
    };

    // TODO:线程安全的MANUAL_Q
    class MANUAL_Q {
    private:
        int head_ = 0, tail_ = 0, counter_ = 0;
        Operation_Logger_Type q[Q_LEN];
        bool isEmpty_;

    public:
        void pop() {
            if (counter_ == 0)
                return;
            head_++;
            head_ %= Q_LEN;
            counter_--;
            if (counter_ == 0)
                isEmpty_ = true;
        }
        Operation_Logger_Type front() { return q[head_]; }
        Operation_Logger_Type back() { return q[tail_]; }
        void clear() {
            head_    = 0;
            tail_    = 0;
            counter_ = 0;
            isEmpty_ = true;
        }
        void push(Operation_Logger_Type op) {
            q[tail_] = op;
            counter_++;
            if (isEmpty_)
                isEmpty_ = false;
            tail_++;
            tail_ %= Q_LEN;
        }
        bool empty() { return isEmpty_; }
        int size() { return counter_; }
    };

    void InitTreeNode(KdNode* root) {
        root->point.x                  = 0.0f;
        root->point.y                  = 0.0f;
        root->point.z                  = 0.0f;
        root->node_range_x[0]          = 0.0f;
        root->node_range_x[1]          = 0.0f;
        root->node_range_y[0]          = 0.0f;
        root->node_range_y[1]          = 0.0f;
        root->node_range_z[0]          = 0.0f;
        root->node_range_z[1]          = 0.0f;
        root->radius_sq                = 0.0f;
        root->division_axis            = 0;
        root->fatherPtr                = nullptr;
        root->leftSonPtr               = nullptr;
        root->rightSonPtr              = nullptr;
        root->TreeSize                 = 0;
        root->invalid_point_num        = 0;
        root->down_del_num             = 0;
        root->point_deleted            = false;
        root->tree_deleted             = false;
        root->need_push_down_to_left   = false;
        root->need_push_down_to_right  = false;
        root->point_downsample_deleted = false;
        root->workingFlag              = false;
    }

    void buildTree(KdNode** root, int leftIndex, int rightIndex, PointVector& Storage) {
        if (leftIndex > rightIndex)
            return;
        *root = new KdNode;
        InitTreeNode(*root);
        int mid      = (leftIndex + rightIndex) >> 1;
        int div_axis = 0;
        int i;
        // Find the best division Axis
        float min_value[3] = {INFINITY, INFINITY, INFINITY};
        float max_value[3] = {-INFINITY, -INFINITY, -INFINITY};
        float dim_range[3] = {0, 0, 0};
        for (i = leftIndex; i <= rightIndex; i++) {
            min_value[0] = std::min(min_value[0], Storage[i].x);
            min_value[1] = std::min(min_value[1], Storage[i].y);
            min_value[2] = std::min(min_value[2], Storage[i].z);
            max_value[0] = std::max(max_value[0], Storage[i].x);
            max_value[1] = std::max(max_value[1], Storage[i].y);
            max_value[2] = std::max(max_value[2], Storage[i].z);
        }
        // Select the longest dimension as division axis
        for (i = 0; i < 3; i++)
            dim_range[i] = max_value[i] - min_value[i];
        for (i = 1; i < 3; i++)
            if (dim_range[i] > dim_range[div_axis])
                div_axis = i;
        // Divide by the division axis and recursively build.

        (*root)->division_axis = div_axis;
        switch (div_axis) {
        case 0:
            std::nth_element(
                begin(Storage) + leftIndex, begin(Storage) + mid, begin(Storage) + rightIndex + 1,
                [](const auto& a, const auto& b) { return a.x < b.x; });
            break;
        case 1:
            std::nth_element(
                begin(Storage) + leftIndex, begin(Storage) + mid, begin(Storage) + rightIndex + 1,
                [](const auto& a, const auto& b) { return a.y < b.y; });
            break;
        case 2:
            std::nth_element(
                begin(Storage) + leftIndex, begin(Storage) + mid, begin(Storage) + rightIndex + 1,
                [](const auto& a, const auto& b) { return a.z < b.z; });
            break;
        default:
            std::nth_element(
                begin(Storage) + leftIndex, begin(Storage) + mid, begin(Storage) + rightIndex + 1,
                [](const auto& a, const auto& b) { return a.x < b.x; });
            break;
        }
        (*root)->point  = Storage[mid];
        KdNode *leftSon = nullptr, *rightSon = nullptr;
        buildTree(&leftSon, leftIndex, mid - 1, Storage);
        buildTree(&rightSon, mid + 1, rightIndex, Storage);
        (*root)->leftSonPtr  = leftSon;
        (*root)->rightSonPtr = rightSon;
        updateStatus((*root));
    }

    void rebuild(KdNode** root) {
        KdNode* fatherPtr;
        if ((*root)->TreeSize >= Multi_Thread_Rebuild_Point_Num) {
            if (rebuild_ptr_mutex_lock.try_lock()) {
                if (rebuildPtr_ == nullptr || ((*root)->TreeSize > (*rebuildPtr_)->TreeSize)) {
                    rebuildPtr_ = root;
                    cv_.notify_all();
                }
                rebuild_ptr_mutex_lock.unlock();
            }
        } else {
            fatherPtr = (*root)->fatherPtr;
            PCL_Storage.clear();
            flatten(*root, PCL_Storage, DELETE_POINTS_REC);
            deleteTreeNode(root);
            buildTree(root, 0, PCL_Storage.size() - 1, PCL_Storage);
            if (*root != nullptr)
                (*root)->fatherPtr = fatherPtr;
            if (*root == root_)
                STATIC_ROOT_NODE->leftSonPtr = *root;
        }
    }

    void deleteTreeNode(KdNode** root) {
        if (*root == nullptr)
            return;
        pushDown(*root);
        deleteTreeNode(&(*root)->leftSonPtr);
        deleteTreeNode(&(*root)->rightSonPtr);
        delete (*root);
        (*root) = nullptr;
    }

    void addByPoint(KdNode** root, PointType point, bool allowRebuild, int fatherAxis) {
        if (*root == nullptr) {
            *root = new KdNode;
            InitTreeNode(*root);
            (*root)->point         = point;
            (*root)->division_axis = (fatherAxis + 1) % 3;
            Update(*root);
            return;
        }
        (*root)->workingFlag = true;
        Operation_Logger_Type add_log;
        add_log.op    = ADD_POINT;
        add_log.point = point;
        pushDown(*root);
        if (((*root)->division_axis == 0 && point.x < (*root)->point.x)
            || ((*root)->division_axis == 1 && point.y < (*root)->point.y)
            || ((*root)->division_axis == 2 && point.z < (*root)->point.z)) {
            if ((rebuildPtr_ == nullptr) || (*root)->leftSonPtr != *rebuildPtr_) {
                addPoint(&(*root)->leftSonPtr, point, allowRebuild, (*root)->division_axis);
            } else {
                workingFlagMutex.lock();
                addPoint(&(*root)->leftSonPtr, point, false, (*root)->division_axis);
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(add_log);
                    rebuild_logger_mutex_lock.unlock();
                }
                workingFlagMutex.unlock();
            }
        } else {
            if ((rebuildPtr_ == nullptr) || (*root)->rightSonPtr != *rebuildPtr_) {
                addPoint(&(*root)->rightSonPtr, point, allowRebuild, (*root)->division_axis);
            } else {
                workingFlagMutex.lock();
                addPoint(&(*root)->rightSonPtr, point, false, (*root)->division_axis);
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(add_log);
                    rebuild_logger_mutex_lock.unlock();
                }
                workingFlagMutex.unlock();
            }
        }
        updateStatus(*root);
        if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root
            && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num)
            rebuildPtr_ = nullptr;
        bool need_rebuild = allowRebuild & criterionCheck((*root));
        if (need_rebuild)
            rebuild(root);
        if ((*root) != nullptr)
            (*root)->workingFlag = false;
    }

    void deleteByPoint(KdNode** root, PointType point, bool allowRebuild) {
        if ((*root) == nullptr || (*root)->tree_deleted)
            return;

        (*root)->workingFlag = true;
        pushDown(root);
        if (samePoint((*root)->point, point) && !(*root)->point_deleted) {
            (*root)->point_deleted = true;
            (*root)->invalid_point_num++;
            if ((*root)->invalid_point_num == (*root)->TreeSize)
                (*root)->tree_deleted = true;
            return;
        }

        Operation_Logger_Type deleteLog;
        struct timespec timeOut;
        deleteLog.op    = DELETE_POINT;
        deleteLog.point = point;
        if (((*root)->division_axis == 0 && point.x < (*root)->point.x)
            || ((*root)->division_axis == 1 && point.y < (*root)->point.y)
            || ((*root)->division_axis == 2 && point.z < (*root)->point.z)) {
            if (rebuildPtr_ == nullptr || (*root)->leftSonPtr != *rebuildPtr_)
                deleteByPoint((*root)->leftSonPtr, point, allowRebuild);
            else {
                workingFlagMutex.lock();
                deleteByPoint((*root)->leftSonPtr, point, false);
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(deleteLog);
                    rebuild_logger_mutex_lock.unlock();
                }
            }
        } else {
            if (rebuildPtr_ == nullptr || (*root)->rightSonPtr != *rebuildPtr_)
                deleteByPoint((*root)->rightSonPtr, point, allowRebuild);
            else {
                workingFlagMutex.lock();
                deleteByPoint((*root)->rightSonPtr, point, false);
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(deleteLog);
                    rebuild_logger_mutex_lock.unlock();
                }
            }
        }
        updateStatus(root);
        if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root
            && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num)
            rebuildPtr_ = nullptr;
        const bool needRebuild = allowRebuild & criterionCheck(root);
        if (needRebuild)
            rebuild(root);
        if ((*root) != nullptr)
            (*root)->workingFlag = false;
    }

    void addByRange(KdNode** root, BoxPointType& boxPoint, bool allowRebuild) {
        if ((*root) == nullptr)
            return;
        (*root)->workingFlag = true;
        pushDown(*root);
        if (boxPoint.vertex_max[0] <= (*root)->node_range_x[0]
            || boxPoint.vertex_min[0] > (*root)->node_range_x[1])
            return;
        if (boxPoint.vertex_max[1] <= (*root)->node_range_y[0]
            || boxPoint.vertex_min[1] > (*root)->node_range_y[1])
            return;
        if (boxPoint.vertex_max[2] <= (*root)->node_range_z[0]
            || boxPoint.vertex_min[2] > (*root)->node_range_z[1])
            return;

        if (boxPoint.vertex_min[0] <= (*root)->node_range_x[0]
            && boxPoint.vertex_max[0] > (*root)->node_range_x[1]
            && boxPoint.vertex_min[1] <= (*root)->node_range_y[0]
            && boxPoint.vertex_max[1] > (*root)->node_range_y[1]
            && boxPoint.vertex_min[2] <= (*root)->node_range_z[0]
            && boxPoint.vertex_max[2] > (*root)->node_range_z[1]) {
            (*root)->tree_deleted            = false || (*root)->tree_downsample_deleted;
            (*root)->point_deleted           = false || (*root)->point_downsample_deleted;
            (*root)->need_push_down_to_left  = true;
            (*root)->need_push_down_to_right = true;
            (*root)->invalid_point_num       = (*root)->down_del_num;
            return;
        }
        if (boxPoint.vertex_min[0] <= (*root)->point.x && boxPoint.vertex_max[0] > (*root)->point.x
            && boxPoint.vertex_min[1] <= (*root)->point.y
            && boxPoint.vertex_max[1] > (*root)->point.y
            && boxPoint.vertex_min[2] <= (*root)->point.z
            && boxPoint.vertex_max[2] > (*root)->point.z)
            (*root)->point_deleted = (*root)->point_downsample_deleted;

        Operation_Logger_Type addBoxLog;
        struct timespec timeOut;
        addBoxLog.op       = ADD_BOX;
        addBoxLog.boxpoint = boxPoint;
        if (rebuildPtr_ == nullptr || (*root)->left_son_ptr != *rebuildPtr_)
            addByRange(&((*root)->leftSonPtr), boxPoint, allowRebuild);
        else {
            workingFlagMutex.lock();
            addByRange(&((*root)->leftSonPtr), boxPoint, false);
            if (rebuild_flag) {
                rebuild_logger_mutex_lock.lock();
                Rebuild_Logger.push(addBoxLog);
                rebuild_logger_mutex_lock.unlock();
            }
            workingFlagMutex.unlock();
        }
        if (rebuildPtr_ == nullptr || (*root)->right_son_ptr != *rebuildPtr_) {
            addByRange(&((*root)->rightSonPtr), boxPoint, allowRebuild);
        } else {
            workingFlagMutex.lock();
            addByRange(&((*root)->rightSonPtr), boxPoint, false);
            if (rebuild_flag) {
                rebuild_logger_mutex_lock.lock();
                Rebuild_Logger.push(addBoxLog);
                rebuild_logger_mutex_lock.unlock();
            }
            workingFlagMutex.unlock();
        }
        updateStatus(*root);
        if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root
            && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num)
            rebuildPtr_ = nullptr;
        const bool needRebuild = allowRebuild & criterionCheck(*root);
        if (needRebuild)
            rebuild(root);
        if ((*root) != nullptr)
            (*root)->workingFlag = false;
    }

    int deleteByRange(KdNode** root, BoxPointType boxPoint, bool allowRebuild, bool isDownSample) {
        if ((*root) == nullptr || (*root)->tree_deleted)
            return 0;

        (*root)->workingFlag = true;
        pushDown(root);
        int tmpCounter = 0;
        if (boxPoint.vertex_max[0] <= (*root)->node_range_x[0]
            || boxPoint.vertex_min[0] > (*root)->node_range_x[1])
            return 0;
        if (boxPoint.vertex_max[1] <= (*root)->node_range_y[0]
            || boxPoint.vertex_min[1] > (*root)->node_range_y[1])
            return 0;
        if (boxPoint.vertex_max[2] <= (*root)->node_range_z[0]
            || boxPoint.vertex_min[2] > (*root)->node_range_z[1])
            return 0;
        if (boxPoint.vertex_min[0] <= (*root)->node_range_x[0]
            && boxPoint.vertex_max[0] > (*root)->node_range_x[1]
            && boxPoint.vertex_min[1] <= (*root)->node_range_y[0]
            && boxPoint.vertex_max[1] > (*root)->node_range_y[1]
            && boxPoint.vertex_min[2] <= (*root)->node_range_z[0]
            && boxPoint.vertex_max[2] > (*root)->node_range_z[1]) {
            (*root)->tree_deleted            = true;
            (*root)->point_deleted           = true;
            (*root)->need_push_down_to_left  = true;
            (*root)->need_push_down_to_right = true;
            tmpCounter                       = (*root)->TreeSize - (*root)->invalid_point_num;
            (*root)->invalid_point_num       = (*root)->TreeSize;
            if (isDownSample) {
                (*root)->tree_downsample_deleted  = true;
                (*root)->point_downsample_deleted = true;
                (*root)->down_del_num             = (*root)->TreeSize;
            }
            return tmpCounter;
        }
        if (!(*root)->point_deleted && boxPoint.vertex_min[0] <= (*root)->point.x
            && boxPoint.vertex_max[0] > (*root)->point.x
            && boxPoint.vertex_min[1] <= (*root)->point.y
            && boxPoint.vertex_max[1] > (*root)->point.y
            && boxPoint.vertex_min[2] <= (*root)->point.z
            && boxPoint.vertex_max[2] > (*root)->point.z) {
            (*root)->point_deleted = true;
            ++tmpCounter;
            if (isDownSample)
                (*root)->point_downsample_deleted = true;
        }
        Operation_Logger_Type deleteBoxLog;
        struct timespec timeOut;
        if (isDownSample)
            deleteBoxLog.op = DOWNSAMPLE_DELETE;
        else
            deleteBoxLog.op = DELETE_BOX;
        deleteBoxLog.boxpoint = boxPoint;
        if (rebuildPtr_ == nullptr || (*root)->leftSonPtr != *rebuildPtr_)
            tmpCounter += deleteByRange((*root)->leftSonPtr, boxPoint, allowRebuild, isDownSample);
        else {
            workingFlagMutex.lock();
            tmpCounter += deleteByRange((*root)->leftSonPtr, boxPoint, false, isDownSample);
            if (rebuild_flag) {
                rebuild_logger_mutex_lock.lock();
                Rebuild_Logger.push(deleteBoxLog);
                rebuild_logger_mutex_lock.unlock();
            }
            workingFlagMutex.unlock();
        }
        if (rebuildPtr_ == nullptr || (*root)->rightSonPtr != *rebuildPtr_)
            tmpCounter += deleteByRange((*root)->rightSonPtr, boxPoint, allowRebuild, isDownSample);
        else {
            workingFlagMutex.lock();
            tmpCounter += deleteByRange((*root)->rightSonPtr, boxPoint, false, isDownSample);
            if (rebuild_flag) {
                rebuild_logger_mutex_lock.lock();
                Rebuild_Logger.push(deleteBoxLog);
                rebuild_logger_mutex_lock.unlock();
            }
            workingFlagMutex.unlock();
        }
        updateStatus(root);
        if (rebuildPtr_ != nullptr && *rebuildPtr_ == *root
            && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num)
            rebuildPtr_ = nullptr;
        bool needRebuild = allowRebuild & criterionCheck(root);
        if (needRebuild)
            rebuild(root);
        if ((*root) != nullptr)
            (*root)->workingFlag = false;
        return tmpCounter;
    }

    int size() {
        int s = 0;
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_) {
            if (root_ != nullptr) {
                return root_->TreeSize;
            } else {
                return 0;
            }
        } else {
            if (workingFlagMutex.try_lock()) {
                s = root_->TreeSize;
                workingFlagMutex.unlock();
                return s;
            } else {
                return Treesize_tmp;
            }
        }
    }

    BoxPointType range() {
        BoxPointType range;
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_) {
            if (root_ != nullptr) {
                range.vertex_min[0] = root_->node_range_x[0];
                range.vertex_min[1] = root_->node_range_y[0];
                range.vertex_min[2] = root_->node_range_z[0];
                range.vertex_max[0] = root_->node_range_x[1];
                range.vertex_max[1] = root_->node_range_y[1];
                range.vertex_max[2] = root_->node_range_z[1];
            } else {
                memset(&range, 0, sizeof(range));
            }
        } else {
            if (workingFlagMutex.try_lock()) {
                range.vertex_min[0] = root_->node_range_x[0];
                range.vertex_min[1] = root_->node_range_y[0];
                range.vertex_min[2] = root_->node_range_z[0];
                range.vertex_max[0] = root_->node_range_x[1];
                range.vertex_max[1] = root_->node_range_y[1];
                range.vertex_max[2] = root_->node_range_z[1];
                workingFlagMutex.unlock();
            } else {
                memset(&range, 0, sizeof(range));
            }
        }
        return range;
    }

    int validNum() {
        int s = 0;
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_) {
            if (root_ != nullptr)
                return root_->TreeSize - root_->invalid_point_num;
            else
                return 0;
        } else {
            if (workingFlagMutex.try_lock()) {
                s = root_->TreeSize - root_->invalid_point_num;
                workingFlagMutex.unlock();
                return s;
            } else
                return -1;
        }
    }

    void rootAlpha(float& alphaBal, float& alphaDel) {
        if (rebuildPtr_ == nullptr || *rebuildPtr_ != root_) {
            alphaBal = root_->alpha_bal;
            alphaDel = root_->alpha_del;
            return;
        } else {
            if (workingFlagMutex.try_lock()) {
                alphaBal = root_->alpha_bal;
                alphaDel = root_->alpha_del;
                workingFlagMutex.unlock();
                return;
            } else {
                alphaBal = alpha_bal_tmp;
                alphaDel = alpha_del_tmp;
                return;
            }
        }
    }

    // TODO: 这里的rebuildPtr_是否需要加锁？进行封装以增加可读性
    void search(KdNode* root, int kNearest, PointType point, MANUAL_HEAP& q, float maxDist) {
        if (root == nullptr || root->tree_deleted)
            return;
        const float curDist    = calcBoxDist(root, point);
        const float maxDistSqr = maxDist * maxDist;
        if (curDist > maxDistSqr)
            return;
        if (root->need_push_down_to_left || root->need_push_down_to_right) {
            if (root->push_down_mutex_lock.try_lock()) {
                pushDown(root);
                root->push_down_mutex_lock.unlock();
            }
        }
        if (!root->point_deleted) {
            const float dist = calcDist(point, root->point);
            if (dist <= maxDistSqr && (q.size() < kNearest || dist < q.top().dist)) {
                if (q.size() >= kNearest)
                    q.pop();
                PointType_CMP currentPoint(root->point, dist);
                q.push(currentPoint);
            }
        }
        const float distLeft  = calcBoxDist(root->leftSonPtr, point);
        const float distRight = calcBoxDist(root->rightSonPtr, point);
        if (q.size() < kNearest || (distLeft < q.top().dist && distRight < q.top().dist)) {
            if (distLeft <= distRight) {
                if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->leftSonPtr)
                    search(root->leftSonPtr, kNearest, point, q, maxDist);
                else {
                    search_flag_mutex.lock();
                    // TODO:使用条件变量,或原子量保护search_mutex_counter
                    while (search_mutex_counter == -1) {
                        search_flag_mutex.unlock();
                        usleep(1);
                        search_flag_mutex.lock();
                    }
                    ++search_mutex_counter;
                    search_flag_mutex.unlock();
                    search(root->leftSonPtr, kNearest, point, q, maxDist);
                    search_flag_mutex.lock();
                    --search_mutex_counter;
                    search_flag_mutex.unlock();
                }
                if (q.size() < kNearest || distRight < q.top().dist) {
                    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->rightSonPtr)
                        search(root->rightSonPtr, kNearest, point, q, maxDist);
                    else {
                        search_flag_mutex.lock();
                        while (search_mutex_counter == -1) {
                            search_flag_mutex.unlock();
                            usleep(1);
                            search_flag_mutex.lock();
                        }
                        ++search_mutex_counter;
                        search_flag_mutex.unlock();
                        search(root->rightSonPtr, kNearest, point, q, maxDist);
                        search_flag_mutex.lock();
                        --search_mutex_counter;
                        search_flag_mutex.unlock();
                    }
                }
            } else {
                if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->rightSonPtr)
                    search(root->rightSonPtr, kNearest, point, q, maxDist);
                else {
                    search_flag_mutex.lock();
                    while (search_mutex_counter == -1) {
                        search_flag_mutex.unlock();
                        usleep(1);
                        search_flag_mutex.lock();
                    }
                    ++search_mutex_counter;
                    search_flag_mutex.unlock();
                    search(root->rightSonPtr, kNearest, point, q, maxDist);
                    search_flag_mutex.lock();
                    --search_mutex_counter;
                    search_flag_mutex.unlock();
                }
                if (q.size() < kNearest || distLeft < q.top().dist)
                    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->leftSonPtr)
                        search(root->leftSonPtr, kNearest, point, q, maxDist);
                    else {
                        search_flag_mutex.lock();
                        while (search_mutex_counter == -1) {
                            search_flag_mutex.unlock();
                            usleep(1);
                            search_flag_mutex.lock();
                        }
                        ++search_mutex_counter;
                        search_flag_mutex.unlock();
                        search(root->leftSonPtr, kNearest, point, q, maxDist);
                        search_flag_mutex.lock();
                        --search_mutex_counter;
                        search_flag_mutex.unlock();
                    }
            }
        } else {
            if (distLeft < q.top().dist) {
                if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->leftSonPtr)
                    search(root->leftSonPtr, kNearest, point, q, maxDist);
                else {
                    search_flag_mutex.lock();
                    while (search_mutex_counter == -1) {
                        search_flag_mutex.unlock();
                        usleep(1);
                        search_flag_mutex.lock();
                    }
                    ++search_mutex_counter;
                    search_flag_mutex.unlock();
                    search(root->leftSonPtr, kNearest, point, q, maxDist);
                    search_flag_mutex.lock();
                    --search_mutex_counter;
                    search_flag_mutex.unlock();
                }
                if (distRight < q.top().dist) {
                    if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->right_son_ptr) {
                        Search(root->right_son_ptr, kNearest, point, q, maxDist);
                    } else {
                        search_flag_mutex.lock();
                        while (search_mutex_counter == -1) {
                            search_flag_mutex.unlock();
                            usleep(1);
                            search_flag_mutex.lock();
                        }
                        ++search_mutex_counter;
                        search_flag_mutex.unlock();
                        Search(root->right_son_ptr, kNearest, point, q, maxDist);
                        search_flag_mutex.lock();
                        search_mutex_counter -= 1;
                        search_flag_mutex.unlock();
                    }
                }
            }
        }
    }

    void searchByRange(KdNode* root, BoxPointType boxpoint, PointVector& storage) {
        if (root == nullptr || root->tree_deleted)
            return;
        pushDown(root);
        if (boxpoint.vertex_max[0] <= root->node_range_x[0]
            || boxpoint.vertex_min[0] > root->node_range_x[1])
            return;
        if (boxpoint.vertex_max[1] <= root->node_range_y[0]
            || boxpoint.vertex_min[1] > root->node_range_y[1])
            return;
        if (boxpoint.vertex_max[2] <= root->node_range_z[0]
            || boxpoint.vertex_min[2] > root->node_range_z[1])
            return;
        if (boxpoint.vertex_min[0] <= root->node_range_x[0]
            && boxpoint.vertex_max[0] > root->node_range_x[1]
            && boxpoint.vertex_min[1] <= root->node_range_y[0]
            && boxpoint.vertex_max[1] > root->node_range_y[1]
            && boxpoint.vertex_min[2] <= root->node_range_z[0]
            && boxpoint.vertex_max[2] > root->node_range_z[1]) {
            flatten(root, storage, NOT_RECORD);
            return;
        }
        if (boxpoint.vertex_min[0] <= root->point.x && boxpoint.vertex_max[0] > root->point.x
            && boxpoint.vertex_min[1] <= root->point.y && boxpoint.vertex_max[1] > root->point.y
            && boxpoint.vertex_min[2] <= root->point.z && boxpoint.vertex_max[2] > root->point.z) {
            if (!root->point_deleted)
                storage.push_back(root->point);
        }
        if ((rebuildPtr_ == nullptr) || root->left_son_ptr != *rebuildPtr_) {
            Search_by_range(root->left_son_ptr, boxpoint, storage);
        } else {
            search_flag_mutex.lock();
            Search_by_range(root->left_son_ptr, boxpoint, storage);
            search_flag_mutex.unlock();
        }
        if ((rebuildPtr_ == nullptr) || root->right_son_ptr != *rebuildPtr_) {
            Search_by_range(root->right_son_ptr, boxpoint, storage);
        } else {
            search_flag_mutex.lock();
            Search_by_range(root->right_son_ptr, boxpoint, storage);
            search_flag_mutex.unlock();
        }
    }

    void searchByRadius(KdNode* root, PointType point, float radius, PointVector& storage) {
        if (root == nullptr || root->tree_deleted)
            return;
        pushDown(root);
        PointType rangeCenter;
        rangeCenter.x    = (root->node_range_x[0] + root->node_range_x[1]) * 0.5;
        rangeCenter.y    = (root->node_range_y[0] + root->node_range_y[1]) * 0.5;
        rangeCenter.z    = (root->node_range_z[0] + root->node_range_z[1]) * 0.5;
        const float dist = std::sqrt(calcDist(point, rangeCenter));
        if (dist > radius + std::sqrt(root->radius_sq))
            return;
        if (dist <= radius - std::sqrt(root->radius_sq)) {
            flatten(root, storage, NOT_RECORD);
            return;
        }
        if (!root->point_deleted && calcDist(root->point, point) <= radius * radius)
            storage.push_back(root->point);
        if (rebuildPtr_ == nullptr || root->leftSonPtr != *rebuildPtr_)
            searchByRadius(root->leftSonPtr, point, radius, storage);
        else {
            search_flag_mutex.lock();
            searchByRadius(root->leftSonPtr, point, radius, storage);
            search_flag_mutex.unlock();
        }
        if (rebuildPtr_ == nullptr || root->rightSonPtr != *rebuildPtr_)
            searchByRadius(root->rightSonPtr, point, radius, storage);
        else {
            search_flag_mutex.lock();
            searchByRadius(root->rightSonPtr, point, radius, storage);
            search_flag_mutex.unlock();
        }
    }

private:
    void runOperation(KdNode** root, Operation_Logger_Type operation) {
        switch (operation.op) {
        case ADD_POINT: addPoint(root, operation.point, false, (*root)->division_axis); break;
        case ADD_BOX: addByRange(root, operation.boxpoint, false); break;
        case DELETE_POINT: deleteByPoint(root, operation.point, false); break;
        case DELETE_BOX: deleteByRange(root, operation.boxpoint, false, false); break;
        case DOWNSAMPLE_DELETE: deleteByRange(root, operation.boxpoint, false, true); break;
        case PUSH_DOWN:
            (*root)->tree_downsample_deleted |= operation.tree_downsample_deleted;
            (*root)->point_downsample_deleted |= operation.tree_downsample_deleted;
            (*root)->tree_deleted  = operation.tree_deleted || (*root)->tree_downsample_deleted;
            (*root)->point_deleted = (*root)->tree_deleted || (*root)->point_downsample_deleted;
            if (operation.tree_downsample_deleted)
                (*root)->down_del_num = (*root)->TreeSize;
            if (operation.tree_deleted)
                (*root)->invalid_point_num = (*root)->TreeSize;
            else
                (*root)->invalid_point_num = (*root)->down_del_num;
            (*root)->need_push_down_to_left  = true;
            (*root)->need_push_down_to_right = true;
            break;
        default: break;
        }
    }

    void updateStatus(KdNode* root) {
        KdNode* leftSonPtr   = root->leftSonPtr;
        KdNode* rightSonPtr  = root->rightSonPtr;
        float tmp_range_x[2] = {INFINITY, -INFINITY};
        float tmp_range_y[2] = {INFINITY, -INFINITY};
        float tmp_range_z[2] = {INFINITY, -INFINITY};

        if (leftSonPtr && rightSonPtr) {
            root->TreeSize          = leftSonPtr->TreeSize + rightSonPtr->TreeSize + 1;
            root->invalid_point_num = leftSonPtr->invalid_point_num + rightSonPtr->invalid_point_num
                                    + (root->point_deleted ? 1 : 0);
            root->down_del_num = leftSonPtr->down_del_num + rightSonPtr->down_del_num
                               + (root->point_downsample_deleted ? 1 : 0);
            root->tree_downsample_deleted = leftSonPtr->tree_downsample_deleted
                                          & rightSonPtr->tree_downsample_deleted
                                          & root->point_downsample_deleted;
            root->tree_deleted =
                leftSonPtr->tree_deleted && rightSonPtr->tree_deleted && root->point_deleted;
            if (root->tree_deleted
                || (!leftSonPtr->tree_deleted && !rightSonPtr->tree_deleted
                    && !root->point_deleted)) {
                tmp_range_x[0] = std::min(
                    std::min(leftSonPtr->node_range_x[0], rightSonPtr->node_range_x[0]),
                    root->point.x);
                tmp_range_x[1] = std::max(
                    std::max(leftSonPtr->node_range_x[1], rightSonPtr->node_range_x[1]),
                    root->point.x);
                tmp_range_y[0] = std::min(
                    std::min(leftSonPtr->node_range_y[0], rightSonPtr->node_range_y[0]),
                    root->point.y);
                tmp_range_y[1] = std::max(
                    std::max(leftSonPtr->node_range_y[1], rightSonPtr->node_range_y[1]),
                    root->point.y);
                tmp_range_z[0] = std::min(
                    std::min(leftSonPtr->node_range_z[0], rightSonPtr->node_range_z[0]),
                    root->point.z);
                tmp_range_z[1] = std::max(
                    std::max(leftSonPtr->node_range_z[1], rightSonPtr->node_range_z[1]),
                    root->point.z);
            } else {
                if (!leftSonPtr->tree_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], leftSonPtr->node_range_x[0]);
                    tmp_range_x[1] = std::max(tmp_range_x[1], leftSonPtr->node_range_x[1]);
                    tmp_range_y[0] = std::min(tmp_range_y[0], leftSonPtr->node_range_y[0]);
                    tmp_range_y[1] = std::max(tmp_range_y[1], leftSonPtr->node_range_y[1]);
                    tmp_range_z[0] = std::min(tmp_range_z[0], leftSonPtr->node_range_z[0]);
                    tmp_range_z[1] = std::max(tmp_range_z[1], leftSonPtr->node_range_z[1]);
                }
                if (!rightSonPtr->tree_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], rightSonPtr->node_range_x[0]);
                    tmp_range_x[1] = std::max(tmp_range_x[1], rightSonPtr->node_range_x[1]);
                    tmp_range_y[0] = std::min(tmp_range_y[0], rightSonPtr->node_range_y[0]);
                    tmp_range_y[1] = std::max(tmp_range_y[1], rightSonPtr->node_range_y[1]);
                    tmp_range_z[0] = std::min(tmp_range_z[0], rightSonPtr->node_range_z[0]);
                    tmp_range_z[1] = std::max(tmp_range_z[1], rightSonPtr->node_range_z[1]);
                }
                if (!root->point_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                    tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                    tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                    tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                    tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                    tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
                }
            }
        } else if (leftSonPtr) {
            root->TreeSize          = leftSonPtr->TreeSize + 1;
            root->invalid_point_num = leftSonPtr->invalid_point_num + (root->point_deleted ? 1 : 0);
            root->down_del_num =
                leftSonPtr->down_del_num + (root->point_downsample_deleted ? 1 : 0);
            root->tree_downsample_deleted =
                leftSonPtr->tree_downsample_deleted & root->point_downsample_deleted;
            root->tree_deleted = leftSonPtr->tree_deleted && root->point_deleted;
            if (root->tree_deleted || (!leftSonPtr->tree_deleted && !root->point_deleted)) {
                tmp_range_x[0] = std::min(leftSonPtr->node_range_x[0], root->point.x);
                tmp_range_x[1] = std::max(leftSonPtr->node_range_x[1], root->point.x);
                tmp_range_y[0] = std::min(leftSonPtr->node_range_y[0], root->point.y);
                tmp_range_y[1] = std::max(leftSonPtr->node_range_y[1], root->point.y);
                tmp_range_z[0] = std::min(leftSonPtr->node_range_z[0], root->point.z);
                tmp_range_z[1] = std::max(leftSonPtr->node_range_z[1], root->point.z);
            } else {
                if (!leftSonPtr->tree_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], leftSonPtr->node_range_x[0]);
                    tmp_range_x[1] = std::max(tmp_range_x[1], leftSonPtr->node_range_x[1]);
                    tmp_range_y[0] = std::min(tmp_range_y[0], leftSonPtr->node_range_y[0]);
                    tmp_range_y[1] = std::max(tmp_range_y[1], leftSonPtr->node_range_y[1]);
                    tmp_range_z[0] = std::min(tmp_range_z[0], leftSonPtr->node_range_z[0]);
                    tmp_range_z[1] = std::max(tmp_range_z[1], leftSonPtr->node_range_z[1]);
                }
                if (!root->point_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                    tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                    tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                    tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                    tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                    tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
                }
            }
        } else if (rightSonPtr) {
            root->TreeSize = rightSonPtr->TreeSize + 1;
            root->invalid_point_num =
                rightSonPtr->invalid_point_num + (root->point_deleted ? 1 : 0);
            root->down_del_num =
                rightSonPtr->down_del_num + (root->point_downsample_deleted ? 1 : 0);
            root->tree_downsample_deleted =
                rightSonPtr->tree_downsample_deleted & root->point_downsample_deleted;
            root->tree_deleted = rightSonPtr->tree_deleted && root->point_deleted;
            if (root->tree_deleted || (!rightSonPtr->tree_deleted && !root->point_deleted)) {
                tmp_range_x[0] = std::min(rightSonPtr->node_range_x[0], root->point.x);
                tmp_range_x[1] = std::max(rightSonPtr->node_range_x[1], root->point.x);
                tmp_range_y[0] = std::min(rightSonPtr->node_range_y[0], root->point.y);
                tmp_range_y[1] = std::max(rightSonPtr->node_range_y[1], root->point.y);
                tmp_range_z[0] = std::min(rightSonPtr->node_range_z[0], root->point.z);
                tmp_range_z[1] = std::max(rightSonPtr->node_range_z[1], root->point.z);
            } else {
                if (!rightSonPtr->tree_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], rightSonPtr->node_range_x[0]);
                    tmp_range_x[1] = std::max(tmp_range_x[1], rightSonPtr->node_range_x[1]);
                    tmp_range_y[0] = std::min(tmp_range_y[0], rightSonPtr->node_range_y[0]);
                    tmp_range_y[1] = std::max(tmp_range_y[1], rightSonPtr->node_range_y[1]);
                    tmp_range_z[0] = std::min(tmp_range_z[0], rightSonPtr->node_range_z[0]);
                    tmp_range_z[1] = std::max(tmp_range_z[1], rightSonPtr->node_range_z[1]);
                }
                if (!root->point_deleted) {
                    tmp_range_x[0] = std::min(tmp_range_x[0], root->point.x);
                    tmp_range_x[1] = std::max(tmp_range_x[1], root->point.x);
                    tmp_range_y[0] = std::min(tmp_range_y[0], root->point.y);
                    tmp_range_y[1] = std::max(tmp_range_y[1], root->point.y);
                    tmp_range_z[0] = std::min(tmp_range_z[0], root->point.z);
                    tmp_range_z[1] = std::max(tmp_range_z[1], root->point.z);
                }
            }
        } else {
            root->TreeSize                = 1;
            root->invalid_point_num       = (root->point_deleted ? 1 : 0);
            root->down_del_num            = (root->point_downsample_deleted ? 1 : 0);
            root->tree_downsample_deleted = root->point_downsample_deleted;
            root->tree_deleted            = root->point_deleted;
            tmp_range_x[0]                = root->point.x;
            tmp_range_x[1]                = root->point.x;
            tmp_range_y[0]                = root->point.y;
            tmp_range_y[1]                = root->point.y;
            tmp_range_z[0]                = root->point.z;
            tmp_range_z[1]                = root->point.z;
        }
        memcpy(root->node_range_x, tmp_range_x, sizeof(tmp_range_x));
        memcpy(root->node_range_y, tmp_range_y, sizeof(tmp_range_y));
        memcpy(root->node_range_z, tmp_range_z, sizeof(tmp_range_z));
        float x_L       = (root->node_range_x[1] - root->node_range_x[0]) * 0.5;
        float y_L       = (root->node_range_y[1] - root->node_range_y[0]) * 0.5;
        float z_L       = (root->node_range_z[1] - root->node_range_z[0]) * 0.5;
        root->radius_sq = x_L * x_L + y_L * y_L + z_L * z_L;
        if (leftSonPtr != nullptr)
            leftSonPtr->fatherPtr = root;
        if (rightSonPtr != nullptr)
            rightSonPtr->fatherPtr = root;
        if (root == root_ && root->TreeSize > 3) {
            KdNode* son_ptr = root->leftSonPtr;
            if (son_ptr == nullptr)
                son_ptr = root->rightSonPtr;
            float tmp_bal   = float(son_ptr->TreeSize) / (root->TreeSize - 1);
            root->alpha_del = float(root->invalid_point_num) / root->TreeSize;
            root->alpha_bal = (tmp_bal >= 0.5 - EPSS) ? tmp_bal : 1 - tmp_bal;
        }
    }

    void pushDown(KdNode* root) {
        if (root == nullptr)
            return;
        Operation_Logger_Type operation;
        operation.op                      = PUSH_DOWN;
        operation.tree_deleted            = root->tree_deleted;
        operation.tree_downsample_deleted = root->tree_downsample_deleted;
        if (root->need_push_down_to_left && root->leftSonPtr != nullptr) {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->leftSonPtr) {
                root->leftSonPtr->tree_downsample_deleted |= root->tree_downsample_deleted;
                root->leftSonPtr->point_downsample_deleted |= root->tree_downsample_deleted;
                root->leftSonPtr->tree_deleted =
                    root->tree_deleted || root->leftSonPtr->tree_downsample_deleted;
                root->leftSonPtr->point_deleted =
                    root->leftSonPtr->tree_deleted || root->leftSonPtr->point_downsample_deleted;
                if (root->tree_downsample_deleted)
                    root->leftSonPtr->down_del_num = root->leftSonPtr->TreeSize;
                if (root->tree_deleted)
                    root->leftSonPtr->invalid_point_num = root->leftSonPtr->TreeSize;
                else
                    root->leftSonPtr->invalid_point_num = root->leftSonPtr->down_del_num;
                root->leftSonPtr->need_push_down_to_left  = true;
                root->leftSonPtr->need_push_down_to_right = true;
                root->need_push_down_to_left              = false;
            } else {
                workingFlagMutex.lock();
                root->leftSonPtr->tree_downsample_deleted |= root->tree_downsample_deleted;
                root->leftSonPtr->point_downsample_deleted |= root->tree_downsample_deleted;
                root->leftSonPtr->tree_deleted =
                    root->tree_deleted || root->leftSonPtr->tree_downsample_deleted;
                root->leftSonPtr->point_deleted =
                    root->leftSonPtr->tree_deleted || root->leftSonPtr->point_downsample_deleted;
                if (root->tree_downsample_deleted)
                    root->leftSonPtr->down_del_num = root->leftSonPtr->TreeSize;
                if (root->tree_deleted)
                    root->leftSonPtr->invalid_point_num = root->leftSonPtr->TreeSize;
                else
                    root->leftSonPtr->invalid_point_num = root->leftSonPtr->down_del_num;
                root->leftSonPtr->need_push_down_to_left  = true;
                root->leftSonPtr->need_push_down_to_right = true;
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(operation);
                    rebuild_logger_mutex_lock.unlock();
                }
                root->need_push_down_to_left = false;
                workingFlagMutex.unlock();
            }
        }
        if (root->need_push_down_to_right && root->rightSonPtr != nullptr) {
            if (rebuildPtr_ == nullptr || *rebuildPtr_ != root->rightSonPtr) {
                root->rightSonPtr->tree_downsample_deleted |= root->tree_downsample_deleted;
                root->rightSonPtr->point_downsample_deleted |= root->tree_downsample_deleted;
                root->rightSonPtr->tree_deleted =
                    root->tree_deleted || root->rightSonPtr->tree_downsample_deleted;
                root->rightSonPtr->point_deleted =
                    root->rightSonPtr->tree_deleted || root->rightSonPtr->point_downsample_deleted;
                if (root->tree_downsample_deleted)
                    root->rightSonPtr->down_del_num = root->rightSonPtr->TreeSize;
                if (root->tree_deleted)
                    root->rightSonPtr->invalid_point_num = root->rightSonPtr->TreeSize;
                else
                    root->rightSonPtr->invalid_point_num = root->rightSonPtr->down_del_num;
                root->rightSonPtr->need_push_down_to_left  = true;
                root->rightSonPtr->need_push_down_to_right = true;
                root->need_push_down_to_right              = false;
            } else {
                workingFlagMutex.lock();
                root->rightSonPtr->tree_downsample_deleted |= root->tree_downsample_deleted;
                root->rightSonPtr->point_downsample_deleted |= root->tree_downsample_deleted;
                root->rightSonPtr->tree_deleted =
                    root->tree_deleted || root->rightSonPtr->tree_downsample_deleted;
                root->rightSonPtr->point_deleted =
                    root->rightSonPtr->tree_deleted || root->rightSonPtr->point_downsample_deleted;
                if (root->tree_downsample_deleted)
                    root->rightSonPtr->down_del_num = root->rightSonPtr->TreeSize;
                if (root->tree_deleted)
                    root->rightSonPtr->invalid_point_num = root->rightSonPtr->TreeSize;
                else
                    root->rightSonPtr->invalid_point_num = root->rightSonPtr->down_del_num;
                root->rightSonPtr->need_push_down_to_left  = true;
                root->rightSonPtr->need_push_down_to_right = true;
                if (rebuild_flag) {
                    rebuild_logger_mutex_lock.lock();
                    Rebuild_Logger.push(operation);
                    rebuild_logger_mutex_lock.unlock();
                }
                root->need_push_down_to_right = false;
                workingFlagMutex.unlock();
            }
        }
    }

    bool criterionCheck(KdNode* root) {
        if (root->TreeSize <= Minimal_Unbalanced_Tree_Size)
            return false;

        float balance_evaluation = 0.0f;
        float delete_evaluation  = 0.0f;
        KdNode* sonPtr           = root->leftSonPtr;
        if (sonPtr == nullptr)
            sonPtr = root->rightSonPtr;
        delete_evaluation  = float(root->invalid_point_num) / root->TreeSize;
        balance_evaluation = float(sonPtr->TreeSize) / (root->TreeSize - 1);
        if (delete_evaluation > delete_criterion_param)
            return true;

        if (balance_evaluation > balance_criterion_param
            || balance_evaluation < 1 - balance_criterion_param)
            return true;

        return false;
    }

    void flatten(KdNode* root, PointVector& Storage, deletePointStorageSet storage_type) {
        if (root == nullptr)
            return;
        pushDown(root);
        if (!root->point_deleted)
            Storage.push_back(root->point);

        flatten(root->leftSonPtr, Storage, storage_type);
        flatten(root->rightSonPtr, Storage, storage_type);
        switch (storage_type) {
        case NOT_RECORD: break;
        case DELETE_POINTS_REC:
            if (root->point_deleted && !root->point_downsample_deleted) {
                Points_deleted.push_back(root->point);
            }
            break;
        case MULTI_THREAD_REC:
            if (root->point_deleted && !root->point_downsample_deleted) {
                Multithread_Points_deleted.push_back(root->point);
            }
            break;
        default: break;
        }
    }

    void rebuildCallBack() {
        bool terminated = false;
        KdNode *fatherPtr, **new_node_ptr;
        terminated = termination_flag.load(std::memory_order::relaxed);
        while (!terminated) {
            rebuild_ptr_mutex_lock.lock();
            workingFlagMutex.lock();
            if (rebuildPtr_ != nullptr) {
                if (!Rebuild_Logger.empty())
                    std::printf("\n\n\n\n\n\n\n\n\n\n\n ERROR!!! \n\n\n\n\n\n\n\n\n");

                rebuild_flag = true;
                if (*rebuildPtr_ == root_) {
                    Treesize_tmp  = root_->TreeSize;
                    Validnum_tmp  = root_->TreeSize - root_->invalid_point_num;
                    alpha_bal_tmp = root_->alpha_bal;
                    alpha_del_tmp = root_->alpha_del;
                }
                KdNode* oldRoot = (*rebuildPtr_);
                fatherPtr       = (*rebuildPtr_)->fatherPtr;
                PointVector().swap(Rebuild_PCL_Storage);
                // Lock Search
                while (search_mutex_counter.load(std::memory_order::relaxed) != 0)
                    usleep(1);

                search_mutex_counter.store(-1, std::memory_order::relaxed);
                // Lock deleted points cache
                points_deleted_rebuild_mutex_lock.lock();
                flatten(*rebuildPtr_, Rebuild_PCL_Storage, MULTI_THREAD_REC);
                // Unlock deleted points cache
                points_deleted_rebuild_mutex_lock.unlock();
                // Unlock Search
                search_mutex_counter.store(0, std::memory_order::relaxed);
                workingFlagMutex.unlock();
                /* Rebuild and update missed operations*/
                Operation_Logger_Type Operation;
                KdNode* new_root_node = nullptr;
                if (int(Rebuild_PCL_Storage.size()) > 0) {
                    buildTree(
                        &new_root_node, 0, Rebuild_PCL_Storage.size() - 1, Rebuild_PCL_Storage);
                    // Rebuild has been done. Updates the blocked operations into the new tree
                    workingFlagMutex.lock();
                    rebuild_logger_mutex_lock.lock();
                    int tmp_counter = 0;
                    while (!Rebuild_Logger.empty()) {
                        Operation      = Rebuild_Logger.front();
                        max_queue_size = std::max(max_queue_size, Rebuild_Logger.size());
                        Rebuild_Logger.pop();
                        rebuild_logger_mutex_lock.unlock();
                        workingFlagMutex.unlock();
                        run_operation(&new_root_node, Operation);
                        tmp_counter++;
                        if (tmp_counter % 10 == 0)
                            usleep(1);
                        workingFlagMutex.lock();
                        rebuild_logger_mutex_lock.lock();
                    }
                    rebuild_logger_mutex_lock.unlock();
                }
                /* Replace to original tree*/
                // pthread_mutex_lock(&workingFlagMutex);
                while (search_mutex_counter.load(std::memory_order::relaxed) != 0)
                    usleep(1);

                search_mutex_counter.store(-1, std::memory_order::relaxed);
                if (fatherPtr->leftSonPtr == *rebuildPtr_) {
                    fatherPtr->leftSonPtr = new_root_node;
                } else if (fatherPtr->rightSonPtr == *rebuildPtr_) {
                    fatherPtr->rightSonPtr = new_root_node;
                } else {
                    throw "Error: Father ptr incompatible with current node\n";
                }
                if (new_root_node != nullptr)
                    new_root_node->fatherPtr = fatherPtr;
                (*rebuildPtr_) = new_root_node;
                int valid_old  = oldRoot->TreeSize - oldRoot->invalid_point_num;
                int valid_new  = new_root_node->TreeSize - new_root_node->invalid_point_num;
                if (fatherPtr == STATIC_ROOT_NODE)
                    root_ = STATIC_ROOT_NODE->leftSonPtr;
                KdNode* update_root = *rebuildPtr_;
                while (update_root != nullptr && update_root != root_) {
                    update_root = update_root->fatherPtr;
                    if (update_root->workingFlag)
                        break;
                    if (update_root == update_root->fatherPtr->leftSonPtr
                        && update_root->fatherPtr->need_push_down_to_left)
                        break;
                    if (update_root == update_root->fatherPtr->rightSonPtr
                        && update_root->fatherPtr->need_push_down_to_right)
                        break;
                    updateStatus(update_root);
                }
                search_mutex_counter.store(0, std::memory_order::relaxed);
                rebuildPtr_ = nullptr;
                workingFlagMutex.unlock();
                rebuild_flag = false;
                /* Delete discarded tree nodes */
                deleteTreeNode(&oldRoot);
            } else
                workingFlagMutex.unlock();

            rebuild_ptr_mutex_lock.unlock();
            terminated = termination_flag.load(std::memory_order::relaxed);
            usleep(100);
        }
        std::printf("[ikd_tree]: rebuild thread terminated normally\n");
    }

    static inline bool samePoint(const PointType& a, const PointType& b) {
        const bool isSame = std::fabs(a.x - b.x) < EPSS && std::fabs(a.y - b.y) < EPSS
                         && std::fabs(a.z - b.z) < EPSS;
        return isSame;
    }

    static inline float calcDist(const PointType& a, const PointType& b) {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    static inline float calcBoxDist(KdNode* node, const PointType& point) {
        if (node == nullptr)
            return INFINITY;
        const float dx =
            std::max({0.0f, node->node_range_x[0] - point.x, point.x - node->node_range_x[1]});
        const float dy =
            std::max({0.0f, node->node_range_y[0] - point.y, point.y - node->node_range_y[1]});
        const float dz =
            std::max({0.0f, node->node_range_z[0] - point.z, point.z - node->node_range_z[1]});
        return dx * dx + dy * dy + dz * dz;
    }

    // Multi-thread Tree Rebuild
    std::atomic<bool> termination_flag = false;
    bool rebuild_flag                  = false;
    PointVector Rebuild_PCL_Storage;
    KdNode** rebuildPtr_                  = nullptr;
    std::atomic<int> search_mutex_counter = 0;
    static void* multi_thread_ptr(void* arg);
    void multi_thread_rebuild();
    void start_thread();
    void stop_thread();
    void run_operation(KdNode** root, Operation_Logger_Type operation);
    // KD Tree Functions and augmented variables
    int Treesize_tmp = 0, Validnum_tmp = 0;
    float alpha_bal_tmp = 0.5, alpha_del_tmp = 0.0;
    float delete_criterion_param  = 0.5f;
    float balance_criterion_param = 0.7f;
    float downsample_size         = 0.2f;
    bool Delete_Storage_Disabled  = false;
    KdNode* STATIC_ROOT_NODE      = nullptr;
    PointVector Points_deleted;
    PointVector Downsample_Storage;
    PointVector Multithread_Points_deleted;
    static bool point_cmp_x(PointType a, PointType b);
    static bool point_cmp_y(PointType a, PointType b);
    static bool point_cmp_z(PointType a, PointType b);

    PointVector PCL_Storage;
    KdNode* root_ = nullptr;
    std::mutex rebuild_ptr_mutex_lock;
    std::mutex workingFlagMutex;
    std::mutex points_deleted_rebuild_mutex_lock;
    std::mutex rebuild_logger_mutex_lock;
    std::mutex search_flag_mutex;
    std::condition_variable cv_;
    MANUAL_Q Rebuild_Logger;
    int max_queue_size = 0;
};
} // namespace KdTree
