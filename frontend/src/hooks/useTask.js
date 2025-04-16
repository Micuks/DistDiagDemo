import { useState, useEffect, useCallback, useRef } from 'react';
import { taskService } from '../services/taskService';
import { message } from 'antd';

// Determine WebSocket URL based on API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = API_BASE_URL.replace(/^http/, 'ws') + '/ws/tasks'; // Construct WebSocket URL

/**
 * Hook for managing task state and operations using WebSockets.
 */
export const useTask = () => {
    const [tasks, setTasks] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isConnected, setIsConnected] = useState(false); // Track WebSocket connection status
    const websocket = useRef(null); // Ref to hold the WebSocket instance

    // Fetch ALL tasks (for initial load)
    const fetchInitialTasks = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            // Use the GET /api/tasks/ endpoint for the initial list
            const allTasks = await taskService.getTasks(); // Assumes getTasks() fetches all now

            // Sort by start time (most recent first)
            const sortedTasks = allTasks.sort((a, b) => {
                const dateA = a.start_time ? new Date(a.start_time) : 0;
                const dateB = b.start_time ? new Date(b.start_time) : 0;
                return dateB - dateA;
            });
            setTasks(sortedTasks);
            console.log("Initial tasks loaded:", sortedTasks.length);
        } catch (err) {
            console.error('Error fetching initial tasks:', err);
            message.error(`Failed to fetch initial tasks: ${err.message}`);
            setError('Failed to fetch initial tasks');
        } finally {
            setLoading(false);
        }
    }, []); // Empty dependency array - runs once

    // --- WebSocket Effect ---
    useEffect(() => {
        console.log('Setting up WebSocket connection to:', WS_URL);
        // Prevent multiple connections if effect runs multiple times quickly
        if (websocket.current && websocket.current.readyState < 2) { // CONNECTING or OPEN
             console.log("WebSocket connection already exists or is connecting.");
             return;
        }

        websocket.current = new WebSocket(WS_URL);
        const ws = websocket.current;

        ws.onopen = () => {
            console.log('WebSocket connection established.');
            setIsConnected(true);
            setError(null); // Clear previous error on successful connection
            // Optional: Fetch initial tasks again on reconnect? Or assume component remount handles it.
        };

        ws.onmessage = (event) => {
            try {
                const updatedTask = JSON.parse(event.data);
                // console.log('WebSocket message received (Task Update):', updatedTask);

                // Update the tasks state immutably
                setTasks(prevTasks => {
                    const existingTaskIndex = prevTasks.findIndex(task => task.id === updatedTask.id);
                    let newTasks;

                    if (existingTaskIndex > -1) {
                        // Update existing task
                        newTasks = [
                            ...prevTasks.slice(0, existingTaskIndex),
                            updatedTask,
                            ...prevTasks.slice(existingTaskIndex + 1),
                        ];
                    } else {
                        // Add new task (likely from creation)
                        newTasks = [updatedTask, ...prevTasks]; // Add to beginning
                    }

                    // Re-sort after update/add to maintain order
                    return newTasks.sort((a, b) => {
                       const dateA = a.start_time ? new Date(a.start_time) : 0;
                       const dateB = b.start_time ? new Date(b.start_time) : 0;
                       return dateB - dateA;
                   });
                });

            } catch (err) {
                console.error('Error processing WebSocket message:', err, 'Data:', event.data);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setError('WebSocket connection error. Check console.'); // Set a general error
            setIsConnected(false);
             // Note: onerror is often followed by onclose
        };

        ws.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            setIsConnected(false);
            if (!event.wasClean) { // Indicate error if closure wasn't clean
                 setError('WebSocket connection closed unexpectedly. Refresh maybe needed.');
            }
            // Optional: Implement reconnection logic here (e.g., with exponential backoff)
            // Be careful not to create infinite loops if the server is down.
        };

        // Cleanup function: close WebSocket on component unmount
        return () => {
            if (ws && ws.readyState === WebSocket.OPEN) { // Check if open before closing
                 console.log('Closing WebSocket connection.');
                 ws.close(1000, "Component unmounted"); // Close cleanly
            }
             websocket.current = null; // Clear the ref
        };

    }, []); // Empty dependency array: setup WebSocket only once on mount

    // --- Initial Data Load Effect ---
     useEffect(() => {
        fetchInitialTasks();
     }, [fetchInitialTasks]); // Depends on the fetch function definition


    // Create a new task - relies on WebSocket for update, no fetch needed here
    const createTask = useCallback(async (taskCreateData) => {
        setLoading(true); // Still show loading for the action itself
        setError(null);
        try {
            // Call the API to create the task
            const createdTask = await taskService.createTask(taskCreateData);
            // No local fetchTasks needed - update will come via WebSocket broadcast
            message.success(`Task "${createdTask.name}" creation request sent (ID: ${createdTask.id}). Status updates via WebSocket.`);
            // The backend broadcast in create_task API will trigger the onmessage handler
            return createdTask; // Return the initial response from API
        } catch (err) {
            console.error('Error creating task:', err);
            message.error(`Failed to create task: ${err.message}`);
            setError(`Failed to create task: ${err.message}`);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []); // No dependency on fetchTasks

    // Stop a task - relies on WebSocket for update, no fetch needed here
    const stopTask = useCallback(async (taskId) => {
        setLoading(true); // Show loading for the action
        setError(null);
        try {
            // Call the API to stop the task
            const stoppedTask = await taskService.stopTask(taskId);
             // No local fetchTasks needed - update will come via WebSocket broadcast
            message.success(`Stop request sent for task ID ${taskId}. Status updates via WebSocket.`);
            // The backend broadcast in stop_task API will trigger the onmessage handler
            return stoppedTask; // Return the initial response from API ('stopping' status usually)
        } catch (err) {
            console.error(`Error stopping task ${taskId}:`, err);
            message.error(`Failed to stop task ${taskId}: ${err.message}`);
            setError(`Failed to stop task ${taskId}: ${err.message}`);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []); // No dependency on fetchTasks


    return {
        tasks,
        loading,
        error,
        isConnected, // Expose connection status to UI if needed
        fetchTasks: fetchInitialTasks, // Expose initial fetch for manual refresh
        createTask,
        stopTask
    };
}; 