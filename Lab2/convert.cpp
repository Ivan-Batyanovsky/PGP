//
// Created by ivanb on 1/13/2021.
//

#include <iostream>
#include <fstream>
#include <string>

enum MyChoice {Forward = 1, Backward, Error} state = Forward;

void dimParser(const std::string & sx, const std::string & sy, int & m, int & n)
{
    std::string temp1, temp2;
    for (int i = sx.size() - 2; i >= 0; i-=2)
    {
        temp1.append(sx.substr(i, 2));
        temp2.append(sy.substr(i, 2));
    }

    m = std::stoi(temp1, nullptr, 16);
    n = std::stoi(temp2, nullptr, 16);
}

MyChoice fromHexToBinary()
{
    std::string input;
    std::string output;

    std::cin >> input >> output;

    std::ifstream fin(input, std::ios::in);
    if (fin.is_open() == false)
    {
        std::cout << "\n Cant read form fin\n";
        return Error;
    }
    else
    {
        std::cout << "It s fine\n";
    }
    std::ofstream fout(output, std::ios::binary | std::ios::out);
    if (fout.is_open() == false)
    {
        std::cout << "\n Cant write to fout\n";
        return Error;
    }
    else
    {
        std::cout << "It s fine\n";
    }

    int M, N;

    std::string strM, strN;
    fin >> strM >> strN;
    dimParser(strM, strN, M, N);
    fout.write((char *) &M, 4);
    fout.write((char *) &N, 4);

    std::cout << M << ' ' << N << std::endl;
    std::string tempStr;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            fin >> tempStr;
            unsigned char c;
            c = std::stoi(tempStr.substr(0, 2), nullptr, 16);
            fout.write((char *) &c, 1);

            c = std::stoi(tempStr.substr(2, 2), nullptr, 16);
            fout.write((char *) &c, 1);

            c = std::stoi(tempStr.substr(4, 2), nullptr, 16);
            fout.write((char *) &c, 1);

            c = std::stoi(tempStr.substr(6, 2), nullptr, 16);
            fout.write((char *) &c, 1);

        }
    }

    fin.close();
    fout.close();

    return Forward;
}
MyChoice fromBinaryToHex()
{
    std::string input;
    std::string output;

    std::cin >> input >> output;

    std::ifstream fin(input, std::ios::in);
    if (fin.is_open() == false)
    {
        std::cout << "\n Cant read form fin\n";
        return Error;
    }
    else
    {
        std::cout << "It s fine\n";
    }
    std::ofstream fout(output, std::ios::binary | std::ios::out);
    if (fout.is_open() == false)
    {
        std::cout << "\n Cant write to fout\n";
        return Error;
    }
    else
    {
        std::cout << "It s fine\n";
    }

    int M, N;

    fin.read((char *) &M, 4);
    fin.read((char *) &N, 4);

    fout << std::hex << std::uppercase;

    fout << M << ' ' << N << std::endl;


    fout.write((char *) &M, 4);
    fout.write((char *) &N, 4);

    std::cout << M << ' ' << N;

    int tempC;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            fin.read((char *) &tempC, 1);
            fout << tempC;
            fin.read((char *) &tempC, 1);
            fout << tempC;
            fin.read((char *) &tempC, 1);
            fout << tempC;
            fin.read((char *) &tempC, 1);
            fout << tempC << ' ';
        }
        fout << std::endl;
    }

    std::cout << "\nA na etom vse2\n";
    fin.close();
    fout.close();

    return Backward;
}
int main()
{
    int st;
    std::cin >> st;

    if      (st == 1) state = Forward;

    else if (st == 2) state = Backward;

    else              state = Error;

    if (state == Forward) state = fromHexToBinary();
    if (state == Backward) state = fromBinaryToHex();
    if (state == Error) std::cout << "Didnt convert properly\n";
    return 0;
}