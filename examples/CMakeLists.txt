add_executable(forecast_demo forecast_demo.cpp)
target_link_libraries(forecast_demo PRIVATE nabia::nabia)
nabia_set_warnings(forecast_demo)
