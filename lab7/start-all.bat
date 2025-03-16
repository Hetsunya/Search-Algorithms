@echo off

:: Запуск backend
echo Starting Backend...
cd backend
start go run main.go
cd ..


:: Запуск frontend
echo Starting Frontend...
cd frontend
start npm start
cd ..

echo All services started.
pause
