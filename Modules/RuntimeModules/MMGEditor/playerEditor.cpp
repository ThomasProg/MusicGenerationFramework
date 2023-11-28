#include "playerEditor.h"

#include "portable-file-dialogs.h"
#include "ConvertingParser.h"

#include "imgui.h"
#include "imgui_internal.h" // Required for some functions
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

bool show_another_window = true;

PlayerEditor::PlayerEditor()
{
    std::string path = std::string(DATASETS_PATH);
    for (char& c : path)
    {
        if (c == '/')
        {
            c = '\\';
        }
    }
    midiPath = sfPath = path;
}

void PlayerEditor::Render()
{
    ImGui::Begin("Player2", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
    //ImGui::Text("Hello from another window!");
    if (ImGui::Button("Play"))
    {

    }
    if (ImGui::Button("Pause"))
    {

    }

    std::string sf = "SoundFont : " + sfPath;

    if (ImGui::Button(sf.c_str()))
    {
        //auto selection = pfd::open_file("Select a file", path).result();
        auto selection = pfd::open_file("Select a file", sfPath).result();

        if (!selection.empty())
        {
            sfPath = selection[0];
            std::cout << "User selected file " << sfPath << "\n";
            player.LoadSoundfont(sfPath.c_str());
        }
    }

    std::string midi = "MIDI : " + midiPath;

    if (ImGui::Button(midi.c_str()))
    {
        //auto selection = pfd::open_file("Select a file", path).result();
        auto selection = pfd::open_file("Select a file", midiPath).result();

        if (!selection.empty())
        {
            midiPath = selection[0];
            std::cout << "User selected file " << midiPath << "\n";

            ConvertingParser parser;
            parser.LoadFromFile(midiPath.c_str());
            player.notesPerTrack = std::move(parser.notesPerTrack);
            music = std::async(std::launch::async, [this]
            { 
                player.Play();
            });
            //player.(midiPath.c_str());
        }
    }

    static int int_value = 0;
    ImGui::VSliderInt("##int", ImVec2(18, 160), &int_value, 0, 5);
    ImGui::SameLine();

    static float values[7] = { 0.0f, 0.60f, 0.35f, 0.9f, 0.70f, 0.20f, 0.0f };
    ImGui::PushID("set1");
    //for (int i = 0; i < 7; i++)
    //{
        //if (i > 0) ImGui::SameLine();

    int i = 1;

    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i / 7.0f, 0.5f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i / 7.0f, 0.6f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i / 7.0f, 0.7f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i / 7.0f, 0.9f, 0.9f));
    ImGui::VSliderFloat("##v", ImVec2(18, 160), &values[i], 0.0f, 1.0f, "");

    if (player.notesPerTrack.size() > 0)
    {
        //float slider_f = 0.5f;
        float div = float(1000 * 60);
        float slider_f = float(player.GetPlayerTime()) / div;
        int slider_i = 50;
        //ImGui::Text("Underlying float value: %f", slider_f);
        ImGuiSliderFlags flags = ImGuiSliderFlags_None;
        if (ImGui::SliderFloat("SliderFloat (0 -> 1)", &slider_f, 0.0f, 1.0f, "%.3f", flags))
        {
            player.SetTime(slider_f * div);
        }
    }

    if (ImGui::IsItemActive() || ImGui::IsItemHovered())
        ImGui::SetTooltip("%.3f", values[i]);
    ImGui::PopStyleColor(4);

    //}
    ImGui::PopID();

    ImGui::End();
}