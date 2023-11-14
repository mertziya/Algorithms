#include <iostream>
#include <vector>
#include <set>
#include <ctime>
#include <cstdlib>
#include <unordered_set>
#include <stack>
#include <algorithm>
#include <iterator>
#include <utility>
#include <chrono>
#include <cassert>


using namespace std;


//*************************************
typedef pair<int, int> Edge;
typedef set<Edge> EdgeSet;

struct GraphPartitionProblem {
    int n;
    EdgeSet edges;
    int k;
};
//*************************************


bool check_connected(int n, const EdgeSet &edges);
void add_random_edge(int n, EdgeSet &edges);
void dfs(int start, const EdgeSet &edges, unordered_set<int> &visited);

using Partition = pair<vector<int>, vector<int>>;
vector<Partition> generate_all_partitions(int n);
bool edge_between_sets(const Edge &edge, const vector<int> &U, const vector<int> &W);
void generate_partitions_recursive(int current_node, int n, vector<int> &U, vector<int> &W, vector<Partition> &result);
int count_edges_between(const unordered_set<int> &A, const unordered_set<int> &B, const set<Edge> &edges);
void reverse_vector(vector<int> &vec);
bool are_all_elements_same(const vector<int> &vec1, const vector<int> &vec2);

GraphPartitionProblem random_graph_partition_problem(int n, float p, int min_k, int max_k);
Partition brute_force_partition(const GraphPartitionProblem &problem);
Partition greedy_partition(const GraphPartitionProblem &problem);

void onlyHeuristic(int n_start, int n_end, int num_instances, float p, int min_k, int max_k);
void heuristicAccuracy(int n_start, int n_end, int num_instances, float p, int min_k, int max_k, int correctOutput);

void printHeuristic(int n, float p, int min_k, int max_k);
void printBrute(int n, float p, int min_k, int max_k);


//********** Unit Testing *************
void test_add_random_edge();
void test_check_connected();
void test_greedy_partition();
void test_edge_between_sets();
void test_dfs();
void test_generate_all_partitions();
void test_generate_partitions_recursive();
void test_reverse_vector();
void test_are_all_elements_same();
void test_random_graph_partition_problem();
void test_brute_force_partition();

void run_unit_tests();
//*************************************





int main() {
    int n_start = 4;
    int n_end = 8;
    int num_instances = 10;

    float p = 0.5; // probability for the presence of an edge between two nodes in the graph.
    int min_k = 1;
    int max_k = 10;
    
    
    //printHeuristic(n_start, p, min_k, max_k);       // --> prints the input and output for Heuristic Algorithm.
    
    //printBrute(n_start, p, min_k, max_k);           // --> prints the input and output for BruteForce Algorithm.

    
    //onlyHeuristic(n_start, n_end, num_instances, p, min_k, max_k);          // ---> THIS FUNCTION CALCULATES THE RUNTIME OF THE
                                                                            //      HEURISTIC ALGORTIHM
    
    
    //heuristicAccuracy(n_start, n_end, num_instances, p, min_k, max_k, 0);   // ---> THIS FUNCTION CALCULATES THE ACCURACY OF
                                                                            //      HEURISTIC ALGORITHM
    
    
    //run_unit_tests();                                                     // ---> THIS FUNCTION IS USED FOR UNIT TESTING
    
    return 0;
}










void printHeuristic(int n, float p, int min_k, int max_k){
    srand((int)time(0));
    GraphPartitionProblem problem = random_graph_partition_problem(n, p, min_k, max_k);
    Partition solution = greedy_partition(problem);

    cout << "Graph G(V, E):" << endl;
    cout << "Nodes (V): " << 2 * problem.n << endl;
    cout << "Edges (E): ";
    for (const Edge &edge : problem.edges) {
        cout << "(" << edge.first << ", " << edge.second << ") ";
    }
    cout << endl;

    cout << "Positive integer k: " << problem.k << endl;

    if (solution.first.empty() && solution.second.empty()) {
        cout << "No valid partition found." << endl;
    } else {
        cout << "The nodes of G can be partitioned into 2 disjoint sets U and W each of size n and such that the total number of distinct edges in E that connect a node u in U to a node w in W is at most k." << endl;

        cout << "Partition U: ";
        for (int u : solution.first) {
            cout << u << " ";
        }
        cout << endl;

        cout << "Partition W: ";
        for (int w : solution.second) {
            cout << w << " ";
        }
        cout << endl;
    }
    
}
void printBrute(int n, float p, int min_k, int max_k){
    srand((int)time(0));
    GraphPartitionProblem problem = random_graph_partition_problem(n, p, min_k, max_k);
    Partition solution = brute_force_partition(problem);

    cout << "Graph G(V, E):" << endl;
    cout << "Nodes (V): " << 2 * problem.n << endl;
    cout << "Edges (E): ";
    for (const Edge &edge : problem.edges) {
        cout << "(" << edge.first << ", " << edge.second << ") ";
    }
    cout << endl;

    cout << "Positive integer k: " << problem.k << endl;

    if (solution.first.empty() && solution.second.empty()) {
        cout << "No valid partition found." << endl;
    } else {
        cout << "The nodes of G can be partitioned into 2 disjoint sets U and W each of size n and such that the total number of distinct edges in E that connect a node u in U to a node w in W is at most k." << endl;

        cout << "Partition U: ";
        for (int u : solution.first) {
            cout << u << " ";
        }
        cout << endl;

        cout << "Partition W: ";
        for (int w : solution.second) {
            cout << w << " ";
        }
        cout << endl;
    }
    
}



void onlyHeuristic(int n_start, int n_end, int num_instances, float p, int min_k, int max_k){
    
    srand((int)time(0));
    
    for (int n = n_start; n <= n_end; ++n) {
        vector<double> execution_times;

        for (int i = 0; i < num_instances; ++i) {
            GraphPartitionProblem problem = random_graph_partition_problem(n, p, min_k, max_k);
            auto start_time = chrono::high_resolution_clock::now();
            Partition solutionGreedy = greedy_partition(problem);
            auto end_time = chrono::high_resolution_clock::now();

            chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
            execution_times.push_back(elapsed_time.count());
        }

        for (const auto &time : execution_times) {
            cout << "n = " << n << ", time = " << time << " milliseconds" << endl;
        }
    }
}


void heuristicAccuracy(int n_start, int n_end, int num_instances, float p, int min_k, int max_k, int correctOutput){
    
    srand((int)time(0));

    for (int n = n_start; n <= n_end; ++n) {
        vector<double> execution_times;

        for (int i = 0; i < num_instances; ++i) {
            GraphPartitionProblem problem = random_graph_partition_problem(n, p, min_k, max_k);
            Partition solutionBrute = brute_force_partition(problem);
            auto start_time = chrono::high_resolution_clock::now();
            Partition solutionGreedy = greedy_partition(problem);
            auto end_time = chrono::high_resolution_clock::now();

            chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
            execution_times.push_back(elapsed_time.count());

            reverse_vector(solutionGreedy.first);
            reverse_vector(solutionGreedy.second);
            
            if (are_all_elements_same(solutionBrute.first, solutionGreedy.first) && are_all_elements_same(solutionBrute.second, solutionGreedy.second)) {
                correctOutput++;
            }
        }

        for (const auto &time : execution_times) {
            cout << "n = " << n << ", time = " << time << " milliseconds\t" << endl;
        }
    }

    cout << "Accuracy: " << (double)correctOutput / (double)((n_end - n_start + 1) * num_instances) *100 << "%"<<endl;
}




GraphPartitionProblem random_graph_partition_problem(int n, float p, int min_k, int max_k) {
    EdgeSet edges;
    for (int i = 0; i < 2 * n; ++i) {
        for (int j = i + 1; j < 2 * n; ++j) {
            if (static_cast<float>(rand()) / RAND_MAX < p) {
                edges.insert(make_pair(i, j));
            }
        }
    }

    while (!check_connected(2 * n, edges)) {
        add_random_edge(2 * n, edges);
    }

    int k = min_k + rand() % (max_k - min_k + 1);

    return {n, edges, k};
}


Partition brute_force_partition(const GraphPartitionProblem &problem) {
    vector<Partition> all_partitions = generate_all_partitions(problem.n);

    for (const Partition &partition : all_partitions) {
        const vector<int> &U = partition.first;
        const vector<int> &W = partition.second;

        int edge_count = 0;
        for (const Edge &edge : problem.edges) {
            if (edge_between_sets(edge, U, W)) {
                edge_count++;
            }
        }

        if (edge_count <= problem.k) {
            return partition;
        }
    }

    return Partition(vector<int>(), vector<int>());
}

Partition greedy_partition(const GraphPartitionProblem &problem) {
    int n = problem.n;
    const set<Edge> &edges = problem.edges;
    int k = problem.k;

    unordered_set<int> U_set, W_set;

    for (int node = 0; node < 2 * n; ++node) {
        int u_edges = count_edges_between(unordered_set<int>{node}, W_set, edges);
        int w_edges = count_edges_between(unordered_set<int>{node}, U_set, edges);

        if (U_set.size() < n && (W_set.size() >= n || u_edges <= w_edges)) {
            U_set.insert(node);
        } else {
            W_set.insert(node);
        }
    }

    int edge_count = count_edges_between(U_set, W_set, edges);
    vector<int> U(U_set.begin(), U_set.end());
    vector<int> W(W_set.begin(), W_set.end());
    if (edge_count <= k) {
        return make_pair(U, W);
    } else {
        return make_pair(vector<int>(), vector<int>());
    }
}

int count_edges_between(const unordered_set<int> &A, const unordered_set<int> &B, const set<Edge> &edges) {
    int count = 0;

    for (const Edge &edge : edges) {
        int u = edge.first;
        int w = edge.second;
        bool u_in_A = A.count(u) > 0;
        bool w_in_A = A.count(w) > 0;
        bool u_in_B = B.count(u) > 0;
        bool w_in_B = B.count(w) > 0;

        if ((u_in_A && w_in_B) || (u_in_B && w_in_A)) {
            count++;
        }
    }

    return count;
}





void generate_partitions_recursive(int current_node, int n, vector<int> &U, vector<int> &W, vector<Partition> &result) {
    if (current_node == 2 * n) {
        if (U.size() == n && W.size() == n) {
            result.push_back(make_pair(U, W));
        }
        return;
    }

    if (U.size() < n) {
        U.push_back(current_node);
        generate_partitions_recursive(current_node + 1, n, U, W, result);
        U.pop_back();
    }

    if (W.size() < n) {
        W.push_back(current_node);
        generate_partitions_recursive(current_node + 1, n, U, W, result);
        W.pop_back();
    }
}

vector<Partition> generate_all_partitions(int n) {
    vector<int> U, W;
    vector<Partition> result;

    generate_partitions_recursive(0, n, U, W, result);

    return result;
}

bool edge_between_sets(const Edge &edge, const vector<int> &U, const vector<int> &W) {
    bool u_in_U = find(U.begin(), U.end(), edge.first) != U.end();
    bool w_in_W = find(W.begin(), W.end(), edge.second) != W.end();

    bool u_in_W = find(W.begin(), W.end(), edge.first) != W.end();
    bool w_in_U = find(U.begin(), U.end(), edge.second) != U.end();

    return (u_in_U && w_in_W) || (u_in_W && w_in_U);
}

void dfs(int start, const EdgeSet &edges, unordered_set<int> &visited) {
    stack<int> node_stack;
    node_stack.push(start);

    while (!node_stack.empty()) {
        int current = node_stack.top();
        node_stack.pop();

        if (visited.count(current) == 0) {
            visited.insert(current);

            for (const Edge &edge : edges) {
                if (edge.first == current && visited.count(edge.second) == 0) {
                    node_stack.push(edge.second);
                } else if (edge.second == current && visited.count(edge.first) == 0) {
                    node_stack.push(edge.first);
                }
            }
        }
    }
}

bool check_connected(int n, const EdgeSet &edges) {
    unordered_set<int> visited;
    dfs(0, edges, visited);
    return visited.size() == n;
}

void add_random_edge(int n, EdgeSet &edges) {
    int u, v;
    do {
        u = rand() % n;
        v = rand() % n;
    } while (u == v || edges.count(make_pair(min(u, v), max(u, v))) > 0);

    edges.insert(make_pair(min(u, v), max(u, v)));
}

void reverse_vector(vector<int> &vec) {
    int size = (int)vec.size();
    for (int i = 0; i < size / 2; ++i) {
        swap(vec[i], vec[size - 1 - i]);
    }
}

bool are_all_elements_same(const vector<int> &vec1, const vector<int> &vec2) {
    if (vec1.empty() && vec2.empty()) {
        return true;
    }

    if (vec1.empty() || vec2.empty()) {
        return false;
    }

    int first_value = vec1[0];

    for (int val : vec1) {
        if (val != first_value) {
            return false;
        }
    }

    for (int val : vec2) {
        if (val != first_value) {
            return false;
        }
    }

    return true;
}



//**************************************** UNIT TESTING *******************************************************
void test_count_edges_between() {
    EdgeSet edges = {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {4, 5}, {5, 6}};

    unordered_set<int> set1 = {0, 1};
    unordered_set<int> set2 = {2, 3};

    // Test case 1: connected sets
    assert(count_edges_between(set1, set2, edges) == 2);

    // Test case 2: disconnected sets
    unordered_set<int> set3 = {4, 5};
    unordered_set<int> set4 = {6};
    assert(count_edges_between(set3, set4, edges) == 1);

    // Add more test cases if necessary
}

void test_greedy_partition() {
    // Test case 1: successful partition
    GraphPartitionProblem problem1 = {
        2,
        {{0, 1}, {1, 2}, {2, 3}, {0, 3}},
        2
    };

    Partition result1 = greedy_partition(problem1);

    unordered_set<int> U_set1(result1.first.begin(), result1.first.end());
    unordered_set<int> W_set1(result1.second.begin(), result1.second.end());

    int edge_count1 = count_edges_between(U_set1, W_set1, problem1.edges);

    assert(edge_count1 <= problem1.k);

    // Test case 2: unsuccessful partition (too many edges)
    GraphPartitionProblem problem2 = {
        2,
        {{0, 1}, {1, 2}, {2, 3}, {0, 3}, {0, 2}},
        1
    };

    Partition result2 = greedy_partition(problem2);
    vector<int> empty_vec;
    assert(result2.first == empty_vec && result2.second == empty_vec);

    // Add more test cases if necessary
}


void test_check_connected() {
    // Test case 1: connected graph
    EdgeSet edges1 = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    assert(check_connected(4, edges1) == true);

    // Test case 2: disconnected graph
    EdgeSet edges2 = {{0, 1}, {2, 3}};
    assert(check_connected(4, edges2) == false);

    // Add more test cases if necessary
}

void test_add_random_edge() {
    int num_nodes = 4;
    EdgeSet edges = {{0, 1}, {1, 2}};

    // Test that the edge set size increases by 1 after calling add_random_edge
    size_t initial_size = edges.size();
    add_random_edge(num_nodes, edges);
    assert(edges.size() == initial_size + 1);

    // Add more test cases if necessary
}

void test_edge_between_sets() {
    Edge edge(1, 2);
    vector<int> U = {0, 1};
    vector<int> W = {2, 3};
    assert(edge_between_sets(edge, U, W) == true);

    edge = {1, 3};
    assert(edge_between_sets(edge, U, W) == true); // This should be true, not false
}


void test_dfs() {
    EdgeSet edges = {{0, 1}, {1, 2}, {2, 3}};
    unordered_set<int> visited;
    dfs(0, edges, visited);
    assert(visited.size() == 4);

    edges = {{0, 1}, {2, 3}};
    visited.clear();
    dfs(0, edges, visited);
    assert(visited.size() == 2);
}

void test_generate_all_partitions() {
    int n = 2;
    vector<Partition> partitions = generate_all_partitions(n);
    assert(partitions.size() == 6);

    // Check if partitions contain expected sets
    set<vector<int>> expected_partitions = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3}
    };

    for (const auto &partition : partitions) {
        const vector<int> &U = partition.first;
        assert(expected_partitions.count(U) == 1);
        expected_partitions.erase(U);
    }
}

void test_generate_partitions_recursive() {
    int n = 2;
    vector<int> U, W;
    vector<Partition> result;
    generate_partitions_recursive(0, n, U, W, result);

    assert(result.size() == 6);

    // Check if result contains expected sets
    set<vector<int>> expected_partitions = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3}
    };

    for (const auto &partition : result) {
        const vector<int> &U = partition.first;
        assert(expected_partitions.count(U) == 1);
        expected_partitions.erase(U);
    }
}

void test_reverse_vector() {
    vector<int> original = {1, 2, 3, 4, 5};
    vector<int> expected = {5, 4, 3, 2, 1};
    reverse_vector(original);
    assert(original == expected);

    original = {6, 7, 8, 9};
    expected = {9, 8, 7, 6};
    reverse_vector(original);
    assert(original == expected);
}

void test_are_all_elements_same() {
    vector<int> vec1 = {1, 1, 1};
    vector<int> vec2 = {1, 1, 1};
    assert(are_all_elements_same(vec1, vec2));

    vec1 = {4, 4, 4};
    vec2 = {4, 4, 7};
    assert(!are_all_elements_same(vec1, vec2));
}


void test_random_graph_partition_problem() {
    int n = 5;
    float p = 0.5;
    int min_k = 1;
    int max_k = 10;

    GraphPartitionProblem problem = random_graph_partition_problem(n, p, min_k, max_k);
    assert(problem.n == n);
    assert(problem.k >= min_k && problem.k <= max_k);

    // Check that the graph is connected
    unordered_set<int> visited;
    dfs(0, problem.edges, visited);
    assert(visited.size() == 2 * n);
}

void test_brute_force_partition() {
    int n = 2;
    EdgeSet edges = {make_pair(0, 2), make_pair(1, 3)};
    int k = 1;
    GraphPartitionProblem problem{n, edges, k};

    Partition result = brute_force_partition(problem);

    unordered_set<int> U_set(result.first.begin(), result.first.end());
    unordered_set<int> W_set(result.second.begin(), result.second.end());

    int edge_count = count_edges_between(U_set, W_set, edges);

    assert(edge_count <= k);
}


void run_unit_tests() {
    // Test check_connected function
    test_check_connected();

    // Test add_random_edge function
    test_add_random_edge();

    // Test dfs function
    test_dfs();

    // Test generate_all_partitions
    test_generate_all_partitions();
    
    // Test edge_between_sets function
    test_edge_between_sets();
    
    // Test generate_partitions_recursive function
    test_generate_partitions_recursive();
    
    // Test count_edges_between function
    test_count_edges_between();

    // Test reverse_vector function
    test_reverse_vector();
    
    // Test are_all_elements_same function
    test_are_all_elements_same();
    
    // Test random_graph_partition_problem function
    test_random_graph_partition_problem();
    
    // Test brute_force_partition function
    test_brute_force_partition();

    // Test greedy_partition function
    test_greedy_partition();
    
}

//***********************************************************************************************************
