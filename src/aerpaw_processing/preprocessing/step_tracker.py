class StepTracker:
    def __init__(self):
        self.steps = []

    def enter_level(self):
        """Go one level deeper (e.g., from 1 to 1.1)."""
        self.steps.append(1)
        return self._format()

    def exit_level(self):
        """Go back up one level (e.g., from 1.1 to 2)."""
        if self.steps:
            self.steps.pop()
            if self.steps:
                self.steps[-1] += 1
        return self._format()

    def next_step(self):
        """Stay at current level, increment number (e.g., 1.1 to 1.2)."""
        if not self.steps:
            self.steps = [1]
        else:
            self.steps[-1] += 1
        return self._format()

    def _format(self):
        """Returns the current step string."""
        return "Step " + ".".join(map(str, self.steps)) + ":"
