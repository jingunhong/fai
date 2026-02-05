#!/bin/bash
# auto-progress.sh - Automatically picks and completes TODO items from CLAUDE.md
#
# Usage: ./scripts/auto-progress.sh [max_items]
#   max_items: Maximum number of items to complete (default: unlimited)
#
# This script runs fully automated with no manual interaction required.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLAUDE_MD="$PROJECT_ROOT/CLAUDE.md"

MAX_ITEMS="${1:-0}"  # 0 means unlimited
COMPLETED=0

# Claude Code command with full permissions (no manual interaction)
CLAUDE_CMD="claude --dangerously-skip-permissions --print"

echo "=== FAI Auto-Progress Script ==="
echo "Project: $PROJECT_ROOT"
echo "Max items: ${MAX_ITEMS:-unlimited}"
echo "Mode: Fully automated (no manual interaction)"
echo ""

# Function to get the first unchecked TODO item
get_next_todo() {
    # Find first line matching "- [ ] `P" pattern and extract priority and description
    grep -n '^\- \[ \] `P[0-9]`' "$CLAUDE_MD" | head -1
}

# Function to extract task description from TODO line
parse_todo() {
    local line="$1"
    # Extract: line_number, priority, task_name, description
    echo "$line" | sed -E 's/^([0-9]+):\- \[ \] `(P[0-9])` ([^:]+): (.+)$/\1|\2|\3|\4/'
}

# Function to run review and refactor
run_review_refactor() {
    echo ""
    echo "----------------------------------------------"
    echo "Running review and refactor..."
    echo "----------------------------------------------"
    echo ""

    REVIEW_PROMPT="Review and refactor the current codebase status:

1. **Code Review:**
   - Check all recently modified files for code quality issues
   - Ensure consistent coding style across the codebase
   - Verify type hints are complete and correct
   - Check for any code duplication that should be refactored

2. **Test Review:**
   - Ensure test coverage is adequate for new code
   - Check that all tests follow project conventions (functions, not classes; use fixtures)
   - Verify mocks are properly set up for external APIs

3. **Refactor if needed:**
   - Fix any issues found during review
   - Keep changes minimal and focused
   - Do NOT add unnecessary abstractions

4. **Validation:**
   - Run \`uv run pytest\` to ensure all tests pass
   - Run \`uv run pre-commit run --all-files\` to check code quality

5. **Commit if changes were made:**
   - Only commit if you made actual changes during refactoring
   - Use commit message: \"refactor: code review improvements\"
   - Push to remote

If everything looks good and no changes are needed, just confirm the codebase is in good shape."

    if ! $CLAUDE_CMD "$REVIEW_PROMPT"; then
        echo ""
        echo "=== Review/refactor step had an issue, continuing anyway ==="
    fi

    echo ""
    echo "----------------------------------------------"
    echo "Review and refactor complete"
    echo "----------------------------------------------"
}

# Main loop
while true; do
    # Check if we've hit the max items limit
    if [[ "$MAX_ITEMS" -gt 0 && "$COMPLETED" -ge "$MAX_ITEMS" ]]; then
        echo ""
        echo "=== Reached max items limit ($MAX_ITEMS) ==="
        break
    fi

    # Get next TODO item
    TODO_LINE=$(get_next_todo)

    if [[ -z "$TODO_LINE" ]]; then
        echo ""
        echo "=== All TODO items completed! ==="
        break
    fi

    # Parse the TODO item
    PARSED=$(parse_todo "$TODO_LINE")
    LINE_NUM=$(echo "$PARSED" | cut -d'|' -f1)
    PRIORITY=$(echo "$PARSED" | cut -d'|' -f2)
    TASK_NAME=$(echo "$PARSED" | cut -d'|' -f3)
    DESCRIPTION=$(echo "$PARSED" | cut -d'|' -f4)

    echo "=============================================="
    echo "[$PRIORITY] $TASK_NAME"
    echo "Description: $DESCRIPTION"
    echo "=============================================="
    echo ""

    # Build the prompt for Claude Code
    PROMPT="Complete the following TODO item from CLAUDE.md Progress section:

**Task:** $TASK_NAME
**Description:** $DESCRIPTION
**Priority:** $PRIORITY

Instructions:
1. Implement the feature/fix as described
2. Write tests for ALL new functionality:
   - Cover happy path and edge cases
   - Mock external APIs (OpenAI, etc.) - never hit real APIs
   - Use pytest fixtures for shared setup
   - Follow test naming: test_<function>_<scenario>
3. Run \`uv run pytest\` - this checks both tests AND coverage (must be ≥80%)
4. If coverage dropped, add more tests until it's back above 80%
5. Run \`uv run pre-commit run --all-files\` to ensure code quality
6. After implementation is complete and all checks pass, update CLAUDE.md:
   - Change the line \`- [ ] \`$PRIORITY\` $TASK_NAME:\` to \`- [x] \`$PRIORITY\` $TASK_NAME:\`
7. Commit all changes with a descriptive message
8. Push to remote

Important: Follow the coding conventions in CLAUDE.md. Keep it simple and focused.
Every new function MUST have corresponding tests."

    # Run Claude Code for implementation
    echo "Starting Claude Code for implementation..."
    echo ""

    if ! $CLAUDE_CMD "$PROMPT"; then
        echo ""
        echo "=== Claude Code exited with error ==="
        echo "Stopping auto-progress. You can re-run to continue."
        exit 1
    fi

    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "=== Completed item $COMPLETED: $TASK_NAME ==="

    # Run review and refactor after each task
    run_review_refactor

    # Small delay before next iteration
    sleep 2
done

echo ""
echo "=== Auto-Progress Summary ==="
echo "Items completed: $COMPLETED"
echo "Done!"
