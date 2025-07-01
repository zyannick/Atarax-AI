from ataraxai.prompt_engine.base_task import BaseTask

class WorkspaceCalendarEventsTask(BaseTask):
    """
    Task to manage calendar events in a workspace.
    """

    def __init__(self, workspace_id: str):
        super().__init__()
        self.task_type = "workspace_calendar_events"
        self.description = "Manage calendar events in the workspace."
        self.workspace_id = workspace_id

