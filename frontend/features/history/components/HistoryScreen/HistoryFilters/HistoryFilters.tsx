"use client";

interface HistoryFiltersProps {
  filter: string;
  onFilterChange: (filter: string) => void;
}

export function HistoryFilters({
  filter,
  onFilterChange,
}: HistoryFiltersProps) {
  return (
    <div className="flex items-center space-x-4">
      <select
        value={filter}
        onChange={e => onFilterChange(e.target.value)}
        className="input w-48"
      >
        <option value="all">All Sessions</option>
        <option value="completed">Completed</option>
        <option value="processing">Processing</option>
      </select>
    </div>
  );
}




