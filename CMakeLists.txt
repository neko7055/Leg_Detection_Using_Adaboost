cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native")


project (leg_detection)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Boost REQUIRED serialization)
find_package(BLAS)


add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)
add_executable(demo ${PROJECT_SOURCE_DIR}/src/demo.cpp)
add_executable(train ${PROJECT_SOURCE_DIR}/src/train.cpp)
add_executable(test ${PROJECT_SOURCE_DIR}/src/test.cpp)


target_include_directories(main
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(demo
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(train
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(test
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(main
    PUBLIC
        Python3::Python
        Python3::Module
        Python3::NumPy
)

target_link_libraries(demo
    PUBLIC
        Python3::Python
        Python3::Module
        Python3::NumPy
        Boost::serialization
)

target_link_libraries(train
    PUBLIC
        Boost::serialization
)

target_link_libraries(test
    PUBLIC
        Boost::serialization
)


if(BLAS_FOUND)
    target_compile_definitions( main PRIVATE EIGEN_USE_BLAS )
    target_compile_definitions( demo PRIVATE EIGEN_USE_BLAS )
    target_compile_definitions( train PRIVATE EIGEN_USE_BLAS )
    target_compile_definitions( test PRIVATE EIGEN_USE_BLAS )
    
    target_include_directories(main
        PUBLIC
            ${BLAS_INCLUDE_DIR}
    )
    target_include_directories(demo
        PUBLIC
            ${BLAS_INCLUDE_DIR}
    )
    target_include_directories(train
        PUBLIC
            ${BLAS_INCLUDE_DIR}
    )
    target_include_directories(test
        PUBLIC
            ${BLAS_INCLUDE_DIR}
    )

    target_link_libraries(main
        PUBLIC
            ${BLAS_LIBRARIES}
    )
    target_link_libraries(demo
        PUBLIC
            ${BLAS_LIBRARIES}
    )
    target_link_libraries(train
        PUBLIC
            ${BLAS_LIBRARIES}
    )
    target_link_libraries(test
        PUBLIC
            ${BLAS_LIBRARIES}
    )

endif(BLAS_FOUND)
