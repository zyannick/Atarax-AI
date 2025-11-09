import {
  createContext,
  useContext,
  useReducer,
  ReactNode,
  useCallback,
  useMemo,
} from "react";
import {
  Project,
  ChatSession,
  Message,
  AppView,
  AppStatus,
  RagSource,
  ModelInfo,
  BenchmarkTest,
  // BenchmarkResult,
  BenchmarkSession,
} from "./types";
import {
  API_STATUS,
  createProject,
  listProjects,
  createChatSession,
  listSessions,
  updateProjectApi,
  deleteProjectApi,
  renameSessionApi,
  deleteSessionApi,
} from "../lib/api";

interface AppState {
  projects: Project[];
  selectedProjectId: string | null;
  sessions: ChatSession[];
  selectedSessionId: string | null;
  messages: Message[];
  isTyping: boolean;
  currentView: AppView;
  appStatus: AppStatus;
  sidebarCollapsed: boolean;
  ragSources: RagSource[];
  models: ModelInfo[];
  searchQuery: string;
  benchmarkTests: BenchmarkTest[];
  benchmarkSessions: BenchmarkSession[];
  activeBenchmarkId: string | null;
}

type AppAction =
  | { type: "SET_CURRENT_VIEW"; payload: AppView }
  | { type: "SET_APP_STATUS"; payload: AppStatus }
  | { type: "TOGGLE_SIDEBAR" }
  | { type: "SET_SIDEBAR_COLLAPSED"; payload: boolean }
  | { type: "SELECT_PROJECT"; payload: string }
  | { type: "SELECT_SESSION"; payload: string }
  | { type: "ADD_PROJECT"; payload: Project }
  | {
      type: "UPDATE_PROJECT";
      payload: { id: string; name: string; description: string };
    }
  | { type: "DELETE_PROJECT"; payload: { id: string } }
  | { type: "ADD_SESSION"; payload: ChatSession }
  | { type: "ADD_SESSIONS"; payload: ChatSession[] }
  | { type: "RENAME_SESSION"; payload: { id: string; title: string } }
  | { type: "DELETE_SESSION"; payload: { id: string } }
  | {
      type: "ADD_MESSAGE";
      payload: {
        sessionId: string;
        content: string;
        role: "user" | "assistant";
        type?: "text" | "image" | "voice" | "video";
        metadata?: any;
      };
    }
  | { type: "SET_TYPING"; payload: boolean }
  | { type: "ADD_RAG_SOURCE"; payload: { path: string } }
  | { type: "REMOVE_RAG_SOURCE"; payload: { id: string } }
  | { type: "SET_SEARCH_QUERY"; payload: string }
  | { type: "ADD_MODEL"; payload: { name: string; size: string } }
  | { type: "UPDATE_MODEL_PROGRESS"; payload: { id: string; progress: number } }
  | { type: "MARK_MODEL_DOWNLOADED"; payload: { id: string } }
  | {
      type: "ADD_BENCHMARK_TEST";
      payload: {
        name: string;
        description: string;
        category: "performance" | "accuracy" | "reasoning" | "memory";
        estimatedDuration: number;
      };
    }
  | {
      type: "ADD_BENCHMARK_SESSION";
      payload: { name: string; modelIds: string[]; testIds: string[] };
    }
  | { type: "SET_ACTIVE_BENCHMARK"; payload: { id: string } }
  | { type: "SET_PROJECTS"; payload: { projects: Project[] } };

const initialState: AppState = {
  projects: [],
  selectedProjectId: null,
  sessions: [],
  selectedSessionId: null,
  messages: [],
  isTyping: false,
  currentView: "chat",
  appStatus: "unlocked",
  sidebarCollapsed: false,
  ragSources: [],
  models: [],
  searchQuery: "",
  benchmarkTests: [],
  benchmarkSessions: [],
  activeBenchmarkId: null,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_CURRENT_VIEW":
      return { ...state, currentView: action.payload };

    case "SET_APP_STATUS":
      return { ...state, appStatus: action.payload };

    case "TOGGLE_SIDEBAR":
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };

    case "SET_SIDEBAR_COLLAPSED":
      return { ...state, sidebarCollapsed: action.payload };

    case "SELECT_PROJECT":
      return { ...state, selectedProjectId: action.payload };

    case "SELECT_SESSION":
      return { ...state, selectedSessionId: action.payload };

    case "ADD_PROJECT": {
      const newProject = action.payload;
      return {
        ...state,
        projects: [...state.projects, newProject],
        selectedProjectId: newProject.id,
      };
    }

    case "UPDATE_PROJECT": {
      const updatedProjects = state.projects.map((project) =>
        project.id === action.payload.id
          ? {
              ...project,
              name: action.payload.name,
              description: action.payload.description,
            }
          : project,
      );
      return {
        ...state,
        projects: updatedProjects,
      };
    }

    case "DELETE_PROJECT": {
      const filteredProjects = state.projects.filter(
        (project) => project.id !== action.payload.id,
      );
      const newSelectedProjectId =
        state.selectedProjectId === action.payload.id
          ? filteredProjects.length > 0
            ? filteredProjects[0].id
            : null
          : state.selectedProjectId;

      const filteredSessions = state.sessions.filter(
        (session) => session.projectId !== action.payload.id,
      );
      const newSelectedSessionId = state.sessions.some(
        (session) =>
          session.id === state.selectedSessionId &&
          session.projectId === action.payload.id,
      )
        ? filteredSessions.length > 0
          ? filteredSessions[0].id
          : null
        : state.selectedSessionId;

      return {
        ...state,
        projects: filteredProjects,
        selectedProjectId: newSelectedProjectId,
        sessions: filteredSessions,
        selectedSessionId: newSelectedSessionId,
      };
    }

    case "ADD_SESSION": {
      const newSession = action.payload;
      return {
        ...state,
        sessions: [...state.sessions, newSession],
        selectedSessionId: newSession.id,
      };
    }

    case "ADD_SESSIONS": {
      const existingSessionIds = new Set(state.sessions.map((s) => s.id));
      const newSessions = action.payload.filter(
        (s) => !existingSessionIds.has(s.id),
      );
      return {
        ...state,
        sessions: [...state.sessions, ...newSessions],
      };
    }

    case "RENAME_SESSION": {
      const updatedSessions = state.sessions.map((session) =>
        session.id === action.payload.id
          ? { ...session, title: action.payload.title }
          : session,
      );
      return {
        ...state,
        sessions: updatedSessions,
      };
    }

    case "DELETE_SESSION": {
      const filteredSessions = state.sessions.filter(
        (session) => session.id !== action.payload.id,
      );
      const newSelectedSessionId =
        state.selectedSessionId === action.payload.id
          ? (
              filteredSessions.find(
                (s) => s.projectId === state.selectedProjectId,
              ) || filteredSessions[0]
            )?.id || null
          : state.selectedSessionId;
      return {
        ...state,
        sessions: filteredSessions,
        selectedSessionId: newSelectedSessionId,
      };
    }

    case "ADD_MESSAGE": {
      const newMessage: Message = {
        id: Date.now().toString(),
        sessionId: action.payload.sessionId,
        content: action.payload.content,
        role: action.payload.role,
        timestamp: new Date(),
        type: action.payload.type || "text",
        metadata: action.payload.metadata,
      };
      return {
        ...state,
        messages: [...state.messages, newMessage],
      };
    }

    case "SET_TYPING":
      return { ...state, isTyping: action.payload };

    case "SET_PROJECTS": {
      const { projects } = action.payload;
      const newSelectedProjectId = projects.length > 0 ? projects[0].id : null;

      return {
        ...state,
        projects: projects,
        selectedProjectId: newSelectedProjectId,
        sessions: [],
        selectedSessionId: null,
        messages: [],
      };
    }

    case "ADD_RAG_SOURCE": {
      const newSource: RagSource = {
        id: Date.now().toString(),
        path: action.payload.path,
        type: "directory",
        addedAt: new Date(),
      };
      return {
        ...state,
        ragSources: [...state.ragSources, newSource],
      };
    }

    case "REMOVE_RAG_SOURCE":
      return {
        ...state,
        ragSources: state.ragSources.filter(
          (source) => source.id !== action.payload.id,
        ),
      };

    case "SET_SEARCH_QUERY":
      return { ...state, searchQuery: action.payload };

    case "ADD_MODEL": {
      const newModel: ModelInfo = {
        id: Date.now().toString(),
        name: action.payload.name,
        size: action.payload.size,
        isDownloaded: false,
        isDownloading: true,
        downloadProgress: 0,
      };
      return {
        ...state,
        models: [...state.models, newModel],
      };
    }

    case "UPDATE_MODEL_PROGRESS":
      return {
        ...state,
        models: state.models.map((model) =>
          model.id === action.payload.id
            ? { ...model, downloadProgress: action.payload.progress }
            : model,
        ),
      };

    case "MARK_MODEL_DOWNLOADED":
      return {
        ...state,
        models: state.models.map((model) =>
          model.id === action.payload.id
            ? {
                ...model,
                isDownloaded: true,
                isDownloading: false,
                downloadProgress: 100,
              }
            : model,
        ),
      };

    case "ADD_BENCHMARK_TEST": {
      const newTest: BenchmarkTest = {
        id: Date.now().toString(),
        name: action.payload.name,
        description: action.payload.description,
        category: action.payload.category,
        estimatedDuration: action.payload.estimatedDuration,
      };
      return {
        ...state,
        benchmarkTests: [...state.benchmarkTests, newTest],
      };
    }

    case "ADD_BENCHMARK_SESSION": {
      const newSession: BenchmarkSession = {
        id: Date.now().toString(),
        name: action.payload.name,
        modelIds: action.payload.modelIds,
        testIds: action.payload.testIds,
        status: "pending",
        progress: 0,
        results: [],
      };
      return {
        ...state,
        benchmarkSessions: [...state.benchmarkSessions, newSession],
        activeBenchmarkId: newSession.id,
      };
    }

    case "SET_ACTIVE_BENCHMARK":
      return { ...state, activeBenchmarkId: action.payload.id };

    default:
      return state;
  }
}

interface AppContextValue extends AppState {
  fetchInitialData: () => Promise<void>;
  fetchSessionsForProject: (projectId: string) => Promise<void>;
  setCurrentView: (view: AppView) => void;
  setAppStatus: (status: AppStatus) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  selectProject: (id: string) => void;
  selectSession: (id: string) => void;
  addProject: (name: string, description: string) => Promise<void>;
  updateProject: (
    id: string,
    name: string,
    description: string,
  ) => Promise<void>;
  deleteProject: (id: string) => Promise<void>;
  addSession: (projectId: string, title?: string) => Promise<void>;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  addMessage: (
    sessionId: string,
    content: string,
    role: "user" | "assistant",
    type?: "text" | "image" | "voice" | "video",
    metadata?: any,
  ) => void;
  setTyping: (typing: boolean) => void;
  addRagSource: (path: string) => void;
  removeRagSource: (id: string) => void;
  setSearchQuery: (query: string) => void;
  addModel: (name: string, size: string) => void;
  updateModelProgress: (id: string, progress: number) => void;
  markModelDownloaded: (id: string) => void;
  addBenchmarkTest: (
    name: string,
    description: string,
    category: "performance" | "accuracy" | "reasoning" | "memory",
    estimatedDuration: number,
  ) => void;
  addBenchmarkSession: (
    name: string,
    modelIds: string[],
    testIds: string[],
  ) => void;
  setActiveBenchmark: (id: string) => void;
  getSessionsByProject: (projectId: string) => ChatSession[];
  getMessagesBySession: (sessionId: string) => Message[];
}


const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const fetchInitialData = useCallback(async () => {
    console.log("Store: Fetching initial project data...");
    try {
      dispatch({ type: "SET_APP_STATUS", payload: "loading" });
      const response: any = await listProjects();
      if (
        response.status === API_STATUS.SUCCESS &&
        Array.isArray(response.projects)
      ) {
        const apiProjects: any[] = response.projects;
        const allProjects: Project[] = apiProjects.map((apiProject) => ({
          id: apiProject.project_id,
          name: apiProject.name,
          description: apiProject.description,
          createdAt: new Date(apiProject.created_at),
          updatedAt: new Date(apiProject.updated_at),
        }));
        console.log(`Store: Fetched ${allProjects.length} projects.`);
        dispatch({
          type: "SET_PROJECTS",
          payload: { projects: allProjects },
        });
        dispatch({ type: "SET_APP_STATUS", payload: "unlocked" });
      } else {
        throw new Error(response.message || "Failed to fetch projects");
      }
    } catch (error) {
      console.error("Failed to fetch initial data:", error);
      dispatch({ type: "SET_APP_STATUS", payload: "error" as AppStatus });
    }
  }, [dispatch]);

  const fetchSessionsForProject = useCallback(
    async (projectId: string) => {
      if (state.sessions.some((s) => s.projectId === projectId)) {
        console.log(`Store: Sessions for project ${projectId} already loaded.`);
        return;
      }
      console.log(`Store: Fetching sessions for project ${projectId}...`);
      try {
        const response = await listSessions(projectId);
        
        if (
          response.status === API_STATUS.SUCCESS &&
          Array.isArray(response.sessions)
        ) {
          const apiSessions: any[] = response.sessions;

          const newSessions: ChatSession[] = apiSessions.map((apiSession) => ({
            id: apiSession.session_id,
            projectId: apiSession.project_id,
            title: apiSession.title,
            createdAt: new Date(apiSession.created_at),
            updatedAt: new Date(apiSession.updated_at),
          }));
          console.log(
            `Store: Fetched ${newSessions.length} sessions for ${projectId}.`,
          );
          dispatch({ type: "ADD_SESSIONS", payload: newSessions });
        } else {
          throw new Error(response.message || "Failed to fetch sessions");
        }
      } catch (error) {
        console.error(
          `Failed to fetch sessions for project ${projectId}:`,
          error,
        );
      }
    },
    [state.sessions, dispatch],
  );

  const addProject = useCallback(
    async (name: string, description: string) => {
      try {
        const response = await createProject(name, description);
        if (response.status === API_STATUS.SUCCESS && response.data) {
          const apiProject = response.data;
          const newProject: Project = {
            id: apiProject.project_id,
            name: apiProject.name,
            description: apiProject.description,
            createdAt: new Date(apiProject.created_at),
            updatedAt: new Date(apiProject.updated_at),
          };
          dispatch({ type: "ADD_PROJECT", payload: newProject });
        } else {
          throw new Error(response.message || "Failed to create project");
        }
      } catch (error) {
        console.error("Failed to add project:", error);
      }
    },
    [dispatch],
  );

  const updateProject = useCallback(
    async (id: string, name: string, description: string) => {
      try {
        const response = await updateProjectApi(id, name, description);
        if (response.status === API_STATUS.SUCCESS) {
          dispatch({
            type: "UPDATE_PROJECT",
            payload: { id, name, description },
          });
        } else {
          throw new Error(response.message || "Failed to update project");
        }
      } catch (error) {
        console.error("Failed to update project:", error);
        throw error;
      }
    },
    [dispatch],
  );

  const deleteProject = useCallback(
    async (id: string) => {
      try {
        const response = await deleteProjectApi(id);
        if (
          response.status === API_STATUS.SUCCESS ||
          (response.data && response.data.task_id)
        ) {
          console.log(`Project deletion initiated/completed for ${id}`);
          dispatch({ type: "DELETE_PROJECT", payload: { id } });
        } else {
          throw new Error(response.message || "Failed to delete project");
        }
      } catch (error) {
        console.error("Failed to delete project:", error);
        throw error;
      }
    },
    [dispatch],
  );

  const addSession = useCallback(
    async (projectId: string, title?: string) => {
      try {
        const response = await createChatSession(projectId, title);
        if (response.status === API_STATUS.SUCCESS && response.data) {
          const apiSession = response.data;
          const newSession: ChatSession = {
            id: apiSession.session_id,
            projectId: apiSession.project_id,
            title: apiSession.title,
            createdAt: new Date(apiSession.created_at),
            updatedAt: new Date(apiSession.updated_at),
          };
          dispatch({ type: "ADD_SESSION", payload: newSession });
        } else {
          throw new Error(response.message || "Failed to create session");
        }
      } catch (error) {
        console.error("Failed to add session:", error);
        throw error;
      }
    },
    [dispatch],
  );

  const renameSession = useCallback(
    async (id: string, title: string) => {
      try {
        const response = await renameSessionApi(id, title);
        if (response.status === API_STATUS.SUCCESS) {
          dispatch({ type: "RENAME_SESSION", payload: { id, title } });
        } else {
          throw new Error(response.message || "Failed to rename session");
        }
      } catch (error) {
        console.error("Failed to rename session:", error);
        throw error;
      }
    },
    [dispatch],
  );

  const deleteSession = useCallback(
    async (id: string) => {
      try {
        const response = await deleteSessionApi(id);
        if (
          response.status === API_STATUS.SUCCESS ||
          (response.data && response.data.task_id)
        ) {
          console.log(`Session deletion initiated/completed for ${id}`);
          dispatch({ type: "DELETE_SESSION", payload: { id } });
        } else {
          throw new Error(response.message || "Failed to delete session");
        }
      } catch (error) {
        console.error("Failed to delete session:", error);
        throw error;
      }
    },
    [dispatch],
  );

  const addMessage = useCallback(
    (
      sessionId: string,
      content: string,
      role: "user" | "assistant",
      type?: "text" | "image" | "voice" | "video",
      metadata?: any,
    ) =>
      dispatch({
        type: "ADD_MESSAGE",
        payload: { sessionId, content, role, type, metadata },
      }),
    [dispatch],
  );

  const setTyping = useCallback(
    (typing: boolean) => dispatch({ type: "SET_TYPING", payload: typing }),
    [dispatch],
  );

  const addRagSource = useCallback(
    (path: string) => dispatch({ type: "ADD_RAG_SOURCE", payload: { path } }),
    [dispatch],
  );

  const removeRagSource = useCallback(
    (id: string) => dispatch({ type: "REMOVE_RAG_SOURCE", payload: { id } }),
    [dispatch],
  );

  const setSearchQuery = useCallback(
    (query: string) => dispatch({ type: "SET_SEARCH_QUERY", payload: query }),
    [dispatch],
  );

  const addModel = useCallback(
    (name: string, size: string) =>
      dispatch({ type: "ADD_MODEL", payload: { name, size } }),
    [dispatch],
  );

  const updateModelProgress = useCallback(
    (id: string, progress: number) =>
      dispatch({ type: "UPDATE_MODEL_PROGRESS", payload: { id, progress } }),
    [dispatch],
  );

  const markModelDownloaded = useCallback(
    (id: string) =>
      dispatch({ type: "MARK_MODEL_DOWNLOADED", payload: { id } }),
    [dispatch],
  );

  const addBenchmarkTest = useCallback(
    (
      name: string,
      description: string,
      category: "performance" | "accuracy" | "reasoning" | "memory",
      estimatedDuration: number,
    ) =>
      dispatch({
        type: "ADD_BENCHMARK_TEST",
        payload: { name, description, category, estimatedDuration },
      }),
    [dispatch],
  );

  const addBenchmarkSession = useCallback(
    (name: string, modelIds: string[], testIds: string[]) =>
      dispatch({
        type: "ADD_BENCHMARK_SESSION",
        payload: { name, modelIds, testIds },
      }),
    [dispatch],
  );

  const setActiveBenchmark = useCallback(
    (id: string) => dispatch({ type: "SET_ACTIVE_BENCHMARK", payload: { id } }),
    [dispatch],
  );

  const setCurrentView = useCallback(
    (view: AppView) => dispatch({ type: "SET_CURRENT_VIEW", payload: view }),
    [dispatch],
  );

  const setAppStatus = useCallback(
    (status: AppStatus) =>
      dispatch({ type: "SET_APP_STATUS", payload: status }),
    [dispatch],
  );

  const toggleSidebar = useCallback(
    () => dispatch({ type: "TOGGLE_SIDEBAR" }),
    [dispatch],
  );

  const setSidebarCollapsed = useCallback(
    (collapsed: boolean) =>
      dispatch({ type: "SET_SIDEBAR_COLLAPSED", payload: collapsed }),
    [dispatch],
  );

  const selectProject = useCallback(
    (id: string) => dispatch({ type: "SELECT_PROJECT", payload: id }),
    [dispatch],
  );

  const selectSession = useCallback(
    (id: string) => dispatch({ type: "SELECT_SESSION", payload: id }),
    [dispatch],
  );

  const getSessionsByProject = useCallback(
    (projectId: string) =>
      state.sessions.filter(
        (session: ChatSession) => session.projectId === projectId,
      ),
    [state.sessions],
  );

  const getMessagesBySession = useCallback(
    (sessionId: string) =>
      state.messages.filter(
        (message: Message) => message.sessionId === sessionId,
      ),
    [state.messages],
  );


  const contextValue = useMemo(
    () => ({
      ...state,
      fetchInitialData,
      fetchSessionsForProject,
      setCurrentView,
      setAppStatus,
      toggleSidebar,
      setSidebarCollapsed,
      selectProject,
      selectSession,
      addProject,
      updateProject,
      deleteProject,
      addSession,
      renameSession,
      deleteSession,
      addMessage,
      setTyping,
      addRagSource,
      removeRagSource,
      setSearchQuery,
      addModel,
      updateModelProgress,
      markModelDownloaded,
      addBenchmarkTest,
      addBenchmarkSession,
      setActiveBenchmark,
      getSessionsByProject,
      getMessagesBySession,
    }),
    [
      state,
      fetchInitialData,
      fetchSessionsForProject,
      setCurrentView,
      setAppStatus,
      toggleSidebar,
      setSidebarCollapsed,
      selectProject,
      selectSession,
      addProject,
      updateProject,
      deleteProject,
      addSession,
      renameSession,
      deleteSession,
      addMessage,
      setTyping,
      addRagSource,
      removeRagSource,
      setSearchQuery,
      addModel,
      updateModelProgress,
      markModelDownloaded,
      addBenchmarkTest,
      addBenchmarkSession,
      setActiveBenchmark,
      getSessionsByProject,
      getMessagesBySession,
    ],
  );

  return (
    <AppContext.Provider value={contextValue}>{children}</AppContext.Provider>
  );
}

export function useAppStore(): AppContextValue {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppStore must be used within an AppProvider");
  }
  return context;
}

